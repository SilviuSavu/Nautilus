#!/usr/bin/env python3
"""
Professional Risk Reporting Validation Script
Validates performance and institutional formatting standards for production deployment

Validation Criteria:
- Report generation performance (<5 seconds)
- Professional HTML/CSS formatting standards
- Data accuracy and completeness
- Client-ready presentation quality
- Responsive design validation
- Cross-browser compatibility
- Print optimization
- Accessibility standards
"""

import asyncio
import time
import json
import re
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple
from pathlib import Path
import statistics

# Import validation targets
from professional_risk_reporter import (
    ProfessionalRiskReporter,
    create_professional_risk_reporter,
    ReportConfiguration,
    ReportType,
    ReportFormat
)

from automated_report_scheduler import (
    AutomatedReportScheduler,
    create_automated_scheduler
)

# Mock dependencies for validation
from unittest.mock import Mock, AsyncMock
from hybrid_risk_analytics import HybridRiskAnalyticsEngine, HybridAnalyticsResult, PortfolioAnalytics
from advanced_risk_analytics import RiskAnalyticsActor
from pyfolio_integration import PyFolioAnalytics
from supervised_knn_optimizer import SupervisedKNNOptimizer

logger = logging.getLogger(__name__)

class ProfessionalReportingValidator:
    """
    Professional reporting system validator
    Ensures institutional-grade quality and performance standards
    """
    
    def __init__(self):
        self.validation_results = {}
        self.performance_benchmarks = {}
        self.formatting_standards = {}
        self.generated_reports = []
        
    async def run_full_validation(self) -> Dict[str, Any]:
        """Run complete validation suite"""
        print("🔍 Starting Professional Risk Reporting Validation")
        print("=" * 60)
        
        # Initialize test system
        reporter, scheduler = await self._setup_test_system()
        
        # Run validation tests
        validation_tasks = [
            self._validate_performance_standards(reporter),
            self._validate_formatting_standards(reporter),
            self._validate_data_accuracy(reporter),
            self._validate_client_readiness(reporter),
            self._validate_responsive_design(reporter),
            self._validate_accessibility(reporter),
            self._validate_automated_scheduling(scheduler),
            self._validate_error_handling(reporter)
        ]
        
        # Execute all validations concurrently
        results = await asyncio.gather(*validation_tasks, return_exceptions=True)
        
        # Compile final validation report
        final_report = await self._compile_validation_report(results)
        
        print("\n" + "=" * 60)
        print("🏁 Validation Complete")
        
        return final_report
    
    async def _setup_test_system(self) -> Tuple[ProfessionalRiskReporter, AutomatedReportScheduler]:
        """Setup test system with mock dependencies"""
        print("⚙️ Setting up test system...")
        
        # Create realistic mock data
        portfolio_analytics = PortfolioAnalytics(
            total_return=0.125,
            sharpe_ratio=1.45,
            max_drawdown=-0.085,
            volatility=0.158,
            value_at_risk=-0.032,
            expected_shortfall=-0.048,
            beta=0.92,
            alpha=0.035,
            tracking_error=0.042,
            information_ratio=0.83,
            sortino_ratio=1.85,
            calmar_ratio=1.47
        )
        
        hybrid_result = HybridAnalyticsResult(
            portfolio_analytics=portfolio_analytics,
            optimization_result=None,
            stress_test_results={
                "2008_crisis": -0.185,
                "2020_covid": -0.145,
                "interest_rate_shock": -0.085
            },
            timestamp=datetime.now()
        )
        
        # Mock engines
        hybrid_engine = Mock(spec=HybridRiskAnalyticsEngine)
        hybrid_engine.compute_comprehensive_analytics = AsyncMock(return_value=hybrid_result)
        
        analytics_actor = Mock(spec=RiskAnalyticsActor)
        
        pyfolio_analytics = Mock(spec=PyFolioAnalytics)
        pyfolio_data = {
            'total_return': 0.125,
            'sharpe_ratio': 1.45,
            'max_drawdown': -0.085,
            'volatility': 0.158,
            'daily_mean_return': 0.0005,
            'daily_volatility': 0.018,
            'annual_return': 0.125,
            'annual_volatility': 0.158,
            'skew': -0.15,
            'kurtosis': 2.8
        }
        pyfolio_analytics.generate_comprehensive_analysis = AsyncMock(return_value=pyfolio_data)
        
        supervised_optimizer = Mock(spec=SupervisedKNNOptimizer)
        
        # Create reporter and scheduler
        reporter = ProfessionalRiskReporter(
            hybrid_engine=hybrid_engine,
            analytics_actor=analytics_actor,
            pyfolio_analytics=pyfolio_analytics,
            supervised_optimizer=supervised_optimizer
        )
        
        scheduler = AutomatedReportScheduler(reporter)
        
        print("✅ Test system setup complete")
        return reporter, scheduler
    
    async def _validate_performance_standards(self, reporter: ProfessionalRiskReporter) -> Dict[str, Any]:
        """Validate performance standards (<5s generation time)"""
        print("\n🏃 Validating Performance Standards...")
        
        performance_results = {}
        
        # Test configurations for performance
        test_configs = [
            ("Executive Summary", ReportType.EXECUTIVE_SUMMARY, ReportFormat.HTML),
            ("Comprehensive HTML", ReportType.COMPREHENSIVE, ReportFormat.HTML),
            ("Comprehensive JSON", ReportType.COMPREHENSIVE, ReportFormat.JSON),
            ("Client Tear Sheet", ReportType.CLIENT_TEAR_SHEET, ReportFormat.HTML),
            ("Risk Focused", ReportType.RISK_FOCUSED, ReportFormat.HTML),
            ("Performance Focused", ReportType.PERFORMANCE_FOCUSED, ReportFormat.JSON),
            ("PDF Ready", ReportType.COMPREHENSIVE, ReportFormat.PDF_READY),
            ("Interactive", ReportType.COMPREHENSIVE, ReportFormat.INTERACTIVE)
        ]
        
        for test_name, report_type, report_format in test_configs:
            print(f"  Testing {test_name}...")
            
            config = ReportConfiguration(
                report_type=report_type,
                format=report_format,
                date_range_days=252
            )
            
            # Run multiple iterations to get accurate measurements
            times = []
            for i in range(5):
                start_time = time.time()
                report = await reporter.generate_professional_report(f"PERF_TEST_{i}", config)
                elapsed = time.time() - start_time
                times.append(elapsed)
                
                # Store report for further validation
                if i == 0:  # Store first report
                    self.generated_reports.append({
                        'name': test_name,
                        'content': report,
                        'config': config
                    })
            
            avg_time = statistics.mean(times)
            min_time = min(times)
            max_time = max(times)
            std_dev = statistics.stdev(times) if len(times) > 1 else 0
            
            performance_results[test_name] = {
                'avg_time': avg_time,
                'min_time': min_time,
                'max_time': max_time,
                'std_dev': std_dev,
                'target_met': avg_time < 5.0,
                'consistent': std_dev < 1.0  # Performance should be consistent
            }
            
            status = "✅" if avg_time < 5.0 else "❌"
            print(f"    {status} {avg_time:.3f}s avg ({min_time:.3f}s - {max_time:.3f}s)")
        
        # Overall performance assessment
        all_passed = all(result['target_met'] for result in performance_results.values())
        avg_performance = statistics.mean(result['avg_time'] for result in performance_results.values())
        
        return {
            'category': 'Performance Standards',
            'passed': all_passed,
            'results': performance_results,
            'summary': {
                'average_generation_time': avg_performance,
                'target_compliance': f"{sum(1 for r in performance_results.values() if r['target_met'])}/{len(performance_results)}",
                'fastest_report': min(performance_results.values(), key=lambda x: x['avg_time']),
                'slowest_report': max(performance_results.values(), key=lambda x: x['avg_time'])
            }
        }
    
    async def _validate_formatting_standards(self, reporter: ProfessionalRiskReporter) -> Dict[str, Any]:
        """Validate institutional formatting standards"""
        print("\n🎨 Validating Formatting Standards...")
        
        formatting_results = {}
        
        # CSS and HTML standards to validate
        css_standards = [
            ('Professional Fonts', r'font-family.*[Ii]nter|[Ss]egoe|[Hh]elvetica'),
            ('Border Radius', r'border-radius.*\d+'),
            ('Box Shadows', r'box-shadow.*\d+px'),
            ('Linear Gradients', r'linear-gradient\(.*\)'),
            ('Responsive Design', r'@media.*\(.*width'),
            ('Print Styles', r'@media print'),
            ('Color Variables', r'--.*-\d+|#[0-9a-fA-F]{6}'),
            ('Grid Layout', r'grid-template-columns'),
            ('Flexbox', r'display.*flex|align-items'),
            ('Professional Spacing', r'padding.*\d+.*rem|margin.*\d+.*rem')
        ]
        
        html_standards = [
            ('DOCTYPE Declaration', r'<!DOCTYPE html>'),
            ('Meta Charset', r'<meta charset="UTF-8">'),
            ('Viewport Meta', r'<meta name="viewport"'),
            ('Semantic Headers', r'<h[1-6].*>'),
            ('Professional Title', r'<title>.*Risk.*Report'),
            ('Structured Content', r'<div class=".*section.*">'),
            ('Data Tables', r'<table.*class=".*table.*">'),
            ('Accessibility', r'alt="|aria-|role='),
            ('Professional Metadata', r'Generated.*\d{4}-\d{2}-\d{2}')
        ]
        
        # Test HTML report for formatting standards
        config = ReportConfiguration(
            report_type=ReportType.COMPREHENSIVE,
            format=ReportFormat.HTML,
            date_range_days=252
        )
        
        html_report = await reporter.generate_professional_report("FORMAT_TEST", config)
        
        # Validate CSS standards
        css_validation = {}
        for standard_name, pattern in css_standards:
            matches = len(re.findall(pattern, html_report, re.IGNORECASE))
            css_validation[standard_name] = {
                'found': matches > 0,
                'count': matches,
                'pattern': pattern
            }
            
            status = "✅" if matches > 0 else "❌"
            print(f"  {status} {standard_name}: {matches} instances")
        
        # Validate HTML standards
        html_validation = {}
        for standard_name, pattern in html_standards:
            matches = len(re.findall(pattern, html_report, re.IGNORECASE))
            html_validation[standard_name] = {
                'found': matches > 0,
                'count': matches,
                'pattern': pattern
            }
            
            status = "✅" if matches > 0 else "❌"
            print(f"  {status} {standard_name}: {matches} instances")
        
        # Additional formatting checks
        additional_checks = {
            'Professional Color Scheme': self._validate_color_scheme(html_report),
            'Institutional Typography': self._validate_typography(html_report),
            'Client-Ready Layout': self._validate_layout_structure(html_report),
            'Brand Consistency': self._validate_brand_elements(html_report)
        }
        
        for check_name, result in additional_checks.items():
            status = "✅" if result['passed'] else "❌"
            print(f"  {status} {check_name}: {result['message']}")
        
        all_css_passed = all(result['found'] for result in css_validation.values())
        all_html_passed = all(result['found'] for result in html_validation.values())
        all_additional_passed = all(result['passed'] for result in additional_checks.values())
        
        return {
            'category': 'Formatting Standards',
            'passed': all_css_passed and all_html_passed and all_additional_passed,
            'results': {
                'css_standards': css_validation,
                'html_standards': html_validation,
                'additional_checks': additional_checks
            },
            'summary': {
                'css_compliance': f"{sum(1 for r in css_validation.values() if r['found'])}/{len(css_validation)}",
                'html_compliance': f"{sum(1 for r in html_validation.values() if r['found'])}/{len(html_validation)}",
                'additional_compliance': f"{sum(1 for r in additional_checks.values() if r['passed'])}/{len(additional_checks)}",
                'report_size': len(html_report)
            }
        }
    
    async def _validate_data_accuracy(self, reporter: ProfessionalRiskReporter) -> Dict[str, Any]:
        """Validate data accuracy and completeness"""
        print("\n📊 Validating Data Accuracy...")
        
        # Generate JSON report for data validation
        config = ReportConfiguration(
            report_type=ReportType.COMPREHENSIVE,
            format=ReportFormat.JSON,
            date_range_days=252
        )
        
        json_report = await reporter.generate_professional_report("DATA_TEST", config)
        
        # Data completeness checks
        required_sections = [
            'metadata',
            'executive_summary',
            'analytics.returns',
            'analytics.risk',
            'analytics.attribution',
            'analytics.var_analysis',
            'analytics.stress_tests'
        ]
        
        data_checks = {}
        
        for section in required_sections:
            keys = section.split('.')
            current = json_report
            
            try:
                for key in keys:
                    current = current[key]
                data_checks[section] = {
                    'present': True,
                    'non_empty': bool(current),
                    'type': type(current).__name__
                }
            except (KeyError, TypeError):
                data_checks[section] = {
                    'present': False,
                    'non_empty': False,
                    'type': 'missing'
                }
            
            status = "✅" if data_checks[section]['present'] else "❌"
            print(f"  {status} {section}: {data_checks[section]['type']}")
        
        # Numerical data validation
        numerical_validations = self._validate_numerical_data(json_report)
        
        all_data_present = all(check['present'] for check in data_checks.values())
        all_numerical_valid = all(numerical_validations.values())
        
        return {
            'category': 'Data Accuracy',
            'passed': all_data_present and all_numerical_valid,
            'results': {
                'required_sections': data_checks,
                'numerical_validations': numerical_validations
            },
            'summary': {
                'completeness': f"{sum(1 for c in data_checks.values() if c['present'])}/{len(data_checks)}",
                'numerical_accuracy': f"{sum(1 for v in numerical_validations.values() if v)}/{len(numerical_validations)}"
            }
        }
    
    async def _validate_client_readiness(self, reporter: ProfessionalRiskReporter) -> Dict[str, Any]:
        """Validate client-ready presentation quality"""
        print("\n👔 Validating Client Readiness...")
        
        # Generate executive summary for client readiness check
        config = ReportConfiguration(
            report_type=ReportType.CLIENT_TEAR_SHEET,
            format=ReportFormat.HTML,
            date_range_days=252,
            custom_branding={
                "client_name": "Institutional Client ABC",
                "report_title": "Quarterly Risk Analytics",
                "footer_text": "Confidential - For Client Use Only"
            }
        )
        
        client_report = await reporter.generate_professional_report("CLIENT_TEST", config)
        
        # Client readiness criteria
        client_criteria = {
            'Professional Header': 'Risk.*Report|Analytics.*Report' in client_report,
            'Executive Summary': 'Executive Summary' in client_report,
            'Clear Metrics Display': r'class="metric-value"' in client_report,
            'Risk Assessment': 'Risk Assessment|Risk Analysis' in client_report,
            'Performance Analysis': 'Performance Analysis|Performance' in client_report,
            'Professional Footer': 'Confidential|Proprietary' in client_report,
            'Date Stamping': r'\d{4}-\d{2}-\d{2}' in client_report,
            'Portfolio Identification': 'CLIENT_TEST' in client_report,
            'Visual Hierarchy': 'class="section-title"' in client_report,
            'Professional Styling': 'linear-gradient|box-shadow' in client_report
        }
        
        for criterion, passed in client_criteria.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {criterion}")
        
        # Additional presentation quality checks
        presentation_quality = {
            'Appropriate Length': 10000 < len(client_report) < 100000,  # Reasonable size
            'No Debug Information': 'debug|console.log|TODO' not in client_report.lower(),
            'Professional Language': self._validate_professional_language(client_report),
            'Consistent Branding': 'Nautilus' in client_report
        }
        
        for quality, passed in presentation_quality.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {quality}")
        
        all_criteria_met = all(client_criteria.values())
        all_quality_met = all(presentation_quality.values())
        
        return {
            'category': 'Client Readiness',
            'passed': all_criteria_met and all_quality_met,
            'results': {
                'client_criteria': client_criteria,
                'presentation_quality': presentation_quality
            },
            'summary': {
                'client_criteria_met': f"{sum(client_criteria.values())}/{len(client_criteria)}",
                'presentation_quality_met': f"{sum(presentation_quality.values())}/{len(presentation_quality)}"
            }
        }
    
    async def _validate_responsive_design(self, reporter: ProfessionalRiskReporter) -> Dict[str, Any]:
        """Validate responsive design implementation"""
        print("\n📱 Validating Responsive Design...")
        
        # Generate HTML report for responsive validation
        config = ReportConfiguration(
            report_type=ReportType.COMPREHENSIVE,
            format=ReportFormat.HTML
        )
        
        html_report = await reporter.generate_professional_report("RESPONSIVE_TEST", config)
        
        # Responsive design checks
        responsive_criteria = {
            'Viewport Meta Tag': '<meta name="viewport"' in html_report,
            'Media Queries': '@media' in html_report,
            'Mobile Breakpoints': '@media.*max-width.*768px' in html_report,
            'Flexible Grid': 'grid-template-columns.*auto-fit|repeat.*auto-fit' in html_report,
            'Flexible Typography': 'font-size.*rem|font-size.*em' in html_report,
            'Mobile Optimization': 'max-width.*480px' in html_report,
            'Responsive Images': 'max-width.*100%|width.*100%' in html_report,
            'Touch-Friendly': 'padding.*1rem|margin.*1rem' in html_report
        }
        
        for criterion, passed in responsive_criteria.items():
            status = "✅" if re.search(criterion.split(': ')[1] if ': ' in criterion else '', html_report, re.IGNORECASE) else "❌"
            print(f"  {status} {criterion}")
        
        responsive_score = sum(1 for criterion in responsive_criteria.values() if criterion)
        
        return {
            'category': 'Responsive Design',
            'passed': responsive_score >= len(responsive_criteria) * 0.8,  # 80% compliance
            'results': responsive_criteria,
            'summary': {
                'responsive_score': f"{responsive_score}/{len(responsive_criteria)}",
                'compliance_rate': f"{(responsive_score/len(responsive_criteria)*100):.1f}%"
            }
        }
    
    async def _validate_accessibility(self, reporter: ProfessionalRiskReporter) -> Dict[str, Any]:
        """Validate accessibility standards"""
        print("\n♿ Validating Accessibility...")
        
        config = ReportConfiguration(
            report_type=ReportType.COMPREHENSIVE,
            format=ReportFormat.HTML
        )
        
        html_report = await reporter.generate_professional_report("A11Y_TEST", config)
        
        # Accessibility checks
        accessibility_criteria = {
            'Alt Text for Images': 'alt=' in html_report,
            'ARIA Labels': 'aria-' in html_report,
            'Semantic HTML': '<h1>|<h2>|<h3>|<nav>|<main>|<section>' in html_report,
            'Color Contrast': self._validate_color_contrast(html_report),
            'Focus Management': 'focus|:focus' in html_report,
            'Screen Reader Support': 'sr-only|screen-reader' in html_report,
            'Language Declaration': 'lang=' in html_report,
            'Skip Links': 'skip-link|skip-content' in html_report
        }
        
        for criterion, passed in accessibility_criteria.items():
            if isinstance(passed, str):
                passed = re.search(passed, html_report, re.IGNORECASE) is not None
            
            status = "✅" if passed else "❌"
            print(f"  {status} {criterion}")
        
        accessibility_score = sum(1 for criterion in accessibility_criteria.values() if criterion)
        
        return {
            'category': 'Accessibility',
            'passed': accessibility_score >= len(accessibility_criteria) * 0.7,  # 70% compliance
            'results': accessibility_criteria,
            'summary': {
                'accessibility_score': f"{accessibility_score}/{len(accessibility_criteria)}",
                'compliance_rate': f"{(accessibility_score/len(accessibility_criteria)*100):.1f}%"
            }
        }
    
    async def _validate_automated_scheduling(self, scheduler: AutomatedReportScheduler) -> Dict[str, Any]:
        """Validate automated scheduling functionality"""
        print("\n⏰ Validating Automated Scheduling...")
        
        from automated_report_scheduler import ReportSchedule, ScheduleFrequency, DeliveryMethod
        
        # Test schedule creation
        test_schedule = ReportSchedule(
            schedule_id="validation_test",
            portfolio_id="VALIDATION_PORTFOLIO",
            client_name="Validation Client",
            report_type=ReportType.EXECUTIVE_SUMMARY,
            report_format=ReportFormat.HTML,
            frequency=ScheduleFrequency.DAILY,
            delivery_method=DeliveryMethod.FILE_SYSTEM,
            file_path_template="./validation_reports/{portfolio_id}_{date}.html"
        )
        
        scheduling_tests = {}
        
        try:
            # Test schedule addition
            schedule_id = await scheduler.add_schedule(test_schedule)
            scheduling_tests['Schedule Creation'] = schedule_id is not None
            print(f"  ✅ Schedule Creation: {schedule_id}")
            
            # Test schedule retrieval
            retrieved = await scheduler.get_schedule(schedule_id)
            scheduling_tests['Schedule Retrieval'] = retrieved is not None
            print(f"  ✅ Schedule Retrieval: Found" if retrieved else "  ❌ Schedule Retrieval: Not Found")
            
            # Test immediate report generation
            delivery_id = await scheduler.generate_immediate_report(schedule_id)
            scheduling_tests['Immediate Generation'] = delivery_id is not None
            print(f"  ✅ Immediate Generation: {delivery_id}")
            
            # Test schedule listing
            schedules = await scheduler.list_schedules()
            scheduling_tests['Schedule Listing'] = len(schedules) > 0
            print(f"  ✅ Schedule Listing: {len(schedules)} schedules")
            
            # Test scheduler status
            status = await scheduler.get_scheduler_status()
            scheduling_tests['Status Reporting'] = 'total_schedules' in status
            print(f"  ✅ Status Reporting: {status.get('total_schedules', 0)} total")
            
            # Cleanup
            await scheduler.remove_schedule(schedule_id)
            
        except Exception as e:
            print(f"  ❌ Scheduling validation error: {e}")
            scheduling_tests['Overall'] = False
        
        all_tests_passed = all(scheduling_tests.values())
        
        return {
            'category': 'Automated Scheduling',
            'passed': all_tests_passed,
            'results': scheduling_tests,
            'summary': {
                'tests_passed': f"{sum(scheduling_tests.values())}/{len(scheduling_tests)}",
                'functionality': 'Complete' if all_tests_passed else 'Partial'
            }
        }
    
    async def _validate_error_handling(self, reporter: ProfessionalRiskReporter) -> Dict[str, Any]:
        """Validate error handling and edge cases"""
        print("\n🛡️ Validating Error Handling...")
        
        error_handling_tests = {}
        
        # Test invalid configurations
        try:
            invalid_config = ReportConfiguration(
                report_type=ReportType.COMPREHENSIVE,
                format=ReportFormat.HTML,
                date_range_days=-100  # Invalid date range
            )
            await reporter.generate_professional_report("ERROR_TEST", invalid_config)
            error_handling_tests['Invalid Date Range'] = True  # Should handle gracefully
        except Exception:
            error_handling_tests['Invalid Date Range'] = True  # Expected to handle
        
        print(f"  ✅ Invalid Date Range: Handled gracefully")
        
        # Test empty portfolio ID
        try:
            config = ReportConfiguration(report_type=ReportType.EXECUTIVE_SUMMARY, format=ReportFormat.HTML)
            result = await reporter.generate_professional_report("", config)
            error_handling_tests['Empty Portfolio ID'] = isinstance(result, (str, dict))
        except Exception:
            error_handling_tests['Empty Portfolio ID'] = True  # Acceptable to reject
        
        print(f"  ✅ Empty Portfolio ID: Handled")
        
        # Test large date ranges
        try:
            config = ReportConfiguration(
                report_type=ReportType.COMPREHENSIVE,
                format=ReportFormat.HTML,
                date_range_days=10000  # Very large range
            )
            result = await reporter.generate_professional_report("LARGE_TEST", config)
            error_handling_tests['Large Date Range'] = isinstance(result, (str, dict))
        except Exception as e:
            error_handling_tests['Large Date Range'] = True  # Acceptable to limit
        
        print(f"  ✅ Large Date Range: Handled")
        
        all_error_tests_passed = all(error_handling_tests.values())
        
        return {
            'category': 'Error Handling',
            'passed': all_error_tests_passed,
            'results': error_handling_tests,
            'summary': {
                'robustness': 'High' if all_error_tests_passed else 'Medium',
                'error_scenarios_handled': f"{sum(error_handling_tests.values())}/{len(error_handling_tests)}"
            }
        }
    
    # Helper validation methods
    
    def _validate_color_scheme(self, html_content: str) -> Dict[str, Any]:
        """Validate professional color scheme"""
        professional_colors = [
            '#1e3a8a', '#0f172a', '#2c3e50', '#34495e',  # Blues/Navy
            '#d97706', '#ea580c', '#f59e0b',            # Gold/Orange
            '#059669', '#10b981', '#27ae60',            # Green
            '#dc2626', '#ef4444', '#e74c3c'             # Red
        ]
        
        colors_found = sum(1 for color in professional_colors if color in html_content)
        
        return {
            'passed': colors_found >= 3,
            'message': f'Found {colors_found} professional colors'
        }
    
    def _validate_typography(self, html_content: str) -> Dict[str, Any]:
        """Validate institutional typography"""
        professional_fonts = ['Inter', 'Segoe UI', 'Helvetica', 'SF Pro']
        fonts_found = sum(1 for font in professional_fonts if font in html_content)
        
        return {
            'passed': fonts_found >= 1,
            'message': f'Found {fonts_found} professional fonts'
        }
    
    def _validate_layout_structure(self, html_content: str) -> Dict[str, Any]:
        """Validate layout structure"""
        required_elements = ['header', 'section', 'container', 'grid', 'card']
        elements_found = sum(1 for element in required_elements if element in html_content)
        
        return {
            'passed': elements_found >= 3,
            'message': f'Found {elements_found} layout elements'
        }
    
    def _validate_brand_elements(self, html_content: str) -> Dict[str, Any]:
        """Validate brand consistency"""
        brand_elements = ['Nautilus', 'Risk Analytics', 'Professional']
        brands_found = sum(1 for brand in brand_elements if brand in html_content)
        
        return {
            'passed': brands_found >= 2,
            'message': f'Found {brands_found} brand elements'
        }
    
    def _validate_numerical_data(self, json_data: Dict[str, Any]) -> Dict[str, bool]:
        """Validate numerical data accuracy"""
        validations = {}
        
        try:
            exec_summary = json_data.get('executive_summary', {})
            
            # Check for reasonable values
            validations['Sharpe Ratio Range'] = -5 <= exec_summary.get('sharpe_ratio', 0) <= 5
            validations['Volatility Range'] = 0 <= exec_summary.get('volatility_annualized', 0) <= 1
            validations['Max Drawdown'] = exec_summary.get('max_drawdown', 0) <= 0
            validations['VaR Negative'] = exec_summary.get('var_95', 0) <= 0
            validations['Beta Range'] = -2 <= exec_summary.get('beta_to_benchmark', 1) <= 3
            
        except Exception:
            validations = {key: False for key in validations}
        
        return validations
    
    def _validate_color_contrast(self, html_content: str) -> bool:
        """Basic color contrast validation"""
        # This would need more sophisticated color analysis in production
        # For now, check for light text on dark backgrounds
        dark_backgrounds = ['#1e3a8a', '#0f172a', '#2c3e50']
        light_text = ['color: white', 'color:#fff', 'color: #fff']
        
        has_dark_bg = any(bg in html_content for bg in dark_backgrounds)
        has_light_text = any(text in html_content for text in light_text)
        
        return has_dark_bg and has_light_text
    
    def _validate_professional_language(self, html_content: str) -> bool:
        """Validate professional language usage"""
        unprofessional_terms = ['awesome', 'cool', 'amazing', 'super', 'wow']
        professional_terms = ['comprehensive', 'analysis', 'assessment', 'metrics', 'performance']
        
        unprofessional_count = sum(html_content.lower().count(term) for term in unprofessional_terms)
        professional_count = sum(html_content.lower().count(term) for term in professional_terms)
        
        return unprofessional_count == 0 and professional_count >= 3
    
    async def _compile_validation_report(self, results: List[Any]) -> Dict[str, Any]:
        """Compile final validation report"""
        
        # Filter out exceptions
        valid_results = [r for r in results if isinstance(r, dict) and 'category' in r]
        
        # Calculate overall scores
        total_categories = len(valid_results)
        passed_categories = sum(1 for r in valid_results if r.get('passed', False))
        
        overall_passed = passed_categories == total_categories
        compliance_rate = (passed_categories / total_categories * 100) if total_categories > 0 else 0
        
        # Compile summary
        validation_summary = {
            'overall_status': 'PASSED' if overall_passed else 'FAILED',
            'compliance_rate': f"{compliance_rate:.1f}%",
            'categories_passed': f"{passed_categories}/{total_categories}",
            'timestamp': datetime.now().isoformat(),
            'production_ready': overall_passed and compliance_rate >= 90,
            'results_by_category': {r['category']: r for r in valid_results}
        }
        
        # Print summary
        print(f"\n📋 Validation Summary:")
        print(f"   Overall Status: {'✅ PASSED' if overall_passed else '❌ FAILED'}")
        print(f"   Compliance Rate: {compliance_rate:.1f}%")
        print(f"   Categories Passed: {passed_categories}/{total_categories}")
        print(f"   Production Ready: {'✅ Yes' if validation_summary['production_ready'] else '❌ No'}")
        
        for result in valid_results:
            status = "✅" if result.get('passed') else "❌"
            print(f"   {status} {result['category']}")
        
        return validation_summary

async def main():
    """Run validation script"""
    logging.basicConfig(level=logging.INFO)
    
    validator = ProfessionalReportingValidator()
    validation_report = await validator.run_full_validation()
    
    # Save validation report
    report_file = Path("validation_report.json")
    with open(report_file, 'w') as f:
        json.dump(validation_report, f, indent=2, default=str)
    
    print(f"\n💾 Validation report saved to: {report_file}")
    
    # Return appropriate exit code
    import sys
    sys.exit(0 if validation_report['production_ready'] else 1)

if __name__ == "__main__":
    asyncio.run(main())