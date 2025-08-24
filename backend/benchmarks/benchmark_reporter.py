"""
Benchmark Reporting System with Comprehensive Analytics
=====================================================

Advanced reporting and analysis system for M4 Max benchmarks:
- Comprehensive performance reports with visualizations
- Before/after comparison analysis
- Performance regression detection and alerting
- Export to various formats (JSON, CSV, HTML, PDF)
- Integration with monitoring systems
- Automated report generation and distribution
- Historical trend analysis and insights
"""

import asyncio
import time
import json
import csv
import statistics
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timedelta
import logging
import os
import base64
from io import StringIO, BytesIO

# Reporting and visualization imports
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logger.warning("Visualization libraries not available")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("Pandas not available")

# Import benchmark result types
from .performance_suite import BenchmarkSuiteResult, BenchmarkResult
from .hardware_validation import HardwareValidationReport, ValidationResult
from .container_benchmarks import ContainerSuiteResult, ContainerBenchmarkResult
from .trading_benchmarks import TradingBenchmarkSuite, TradingBenchmarkResult
from .ai_benchmarks import AIBenchmarkSuite, AIBenchmarkResult
from .stress_tests import StressTestSuiteResult, StressTestResult

logger = logging.getLogger(__name__)

@dataclass
class ReportConfig:
    """Report configuration"""
    include_visualizations: bool = True
    include_raw_data: bool = True
    include_recommendations: bool = True
    include_historical_comparison: bool = True
    export_formats: List[str] = None
    output_directory: str = "benchmark_reports"
    
    def __post_init__(self):
        if self.export_formats is None:
            self.export_formats = ["html", "json"]

@dataclass
class BenchmarkReport:
    """Comprehensive benchmark report"""
    report_id: str
    timestamp: datetime
    system_info: Dict[str, Any]
    performance_summary: Dict[str, Any]
    hardware_validation: Optional[Dict[str, Any]] = None
    container_performance: Optional[Dict[str, Any]] = None
    trading_performance: Optional[Dict[str, Any]] = None
    ai_performance: Optional[Dict[str, Any]] = None
    stress_test_results: Optional[Dict[str, Any]] = None
    regression_analysis: Optional[Dict[str, Any]] = None
    recommendations: List[str] = None
    visualizations: Dict[str, str] = None  # Base64 encoded images
    
    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []
        if self.visualizations is None:
            self.visualizations = {}

class BenchmarkReporter:
    """
    Comprehensive benchmark reporting system with analytics
    """
    
    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig()
        
        # Create output directory
        os.makedirs(self.config.output_directory, exist_ok=True)
        
        # Historical data storage
        self.historical_data_file = os.path.join(self.config.output_directory, "historical_data.json")
        self.historical_data = self._load_historical_data()
        
        # Report templates
        self.html_template = self._get_html_template()
        
        # Regression thresholds
        self.regression_thresholds = {
            "performance_degradation": 5.0,  # 5% slower
            "accuracy_drop": 2.0,  # 2% accuracy drop
            "throughput_reduction": 10.0,  # 10% throughput reduction
            "latency_increase": 20.0,  # 20% latency increase
            "error_rate_increase": 1.0  # 1% error rate increase
        }
    
    def _load_historical_data(self) -> List[Dict[str, Any]]:
        """Load historical benchmark data"""
        try:
            if os.path.exists(self.historical_data_file):
                with open(self.historical_data_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load historical data: {e}")
        
        return []
    
    def _save_historical_data(self, report_data: Dict[str, Any]):
        """Save benchmark data to historical records"""
        try:
            # Add timestamp and summary to historical data
            historical_entry = {
                "timestamp": report_data["timestamp"],
                "report_id": report_data["report_id"],
                "performance_summary": report_data.get("performance_summary", {}),
                "system_info": report_data.get("system_info", {}),
                "overall_scores": self._extract_overall_scores(report_data)
            }
            
            self.historical_data.append(historical_entry)
            
            # Keep only last 100 entries
            if len(self.historical_data) > 100:
                self.historical_data = self.historical_data[-100:]
            
            with open(self.historical_data_file, 'w') as f:
                json.dump(self.historical_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save historical data: {e}")
    
    def _extract_overall_scores(self, report_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract overall performance scores from report"""
        scores = {}
        
        # Performance benchmark scores
        if "performance_summary" in report_data:
            perf = report_data["performance_summary"]
            scores["avg_latency_ms"] = perf.get("overall_avg_latency_ms", 0)
            scores["avg_throughput"] = perf.get("avg_throughput", 0)
        
        # Hardware validation score
        if "hardware_validation" in report_data:
            hw = report_data["hardware_validation"]
            scores["hardware_score"] = hw.get("performance_score", 0)
        
        # Trading performance scores
        if "trading_performance" in report_data:
            trading = report_data["trading_performance"]
            perf_summary = trading.get("performance_summary", {})
            scores["trading_avg_latency_ms"] = perf_summary.get("overall_avg_latency_ms", 0)
        
        # AI performance scores
        if "ai_performance" in report_data:
            ai = report_data["ai_performance"]
            perf_summary = ai.get("model_performance_summary", {})
            scores["ai_avg_latency_ms"] = perf_summary.get("overall_avg_latency_ms", 0)
            scores["ai_avg_accuracy"] = perf_summary.get("overall_avg_accuracy", 0)
        
        # Stress test scores
        if "stress_test_results" in report_data:
            stress = report_data["stress_test_results"]
            scores["reliability_score"] = stress.get("overall_reliability_score", 0)
            scores["stability_score"] = stress.get("system_stability_score", 0)
        
        return scores
    
    async def generate_comprehensive_report(
        self,
        performance_results: Optional[BenchmarkSuiteResult] = None,
        hardware_results: Optional[HardwareValidationReport] = None,
        container_results: Optional[ContainerSuiteResult] = None,
        trading_results: Optional[TradingBenchmarkSuite] = None,
        ai_results: Optional[AIBenchmarkSuite] = None,
        stress_results: Optional[StressTestSuiteResult] = None
    ) -> BenchmarkReport:
        """
        Generate comprehensive benchmark report
        """
        logger.info("Generating comprehensive benchmark report")
        
        # Generate unique report ID
        timestamp = datetime.now()
        report_id = f"benchmark_report_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Collect system information
        system_info = self._collect_system_info()
        
        # Initialize report
        report = BenchmarkReport(
            report_id=report_id,
            timestamp=timestamp,
            system_info=system_info,
            performance_summary={}
        )
        
        # Process each benchmark suite
        if performance_results:
            report.performance_summary = self._process_performance_results(performance_results)
        
        if hardware_results:
            report.hardware_validation = self._process_hardware_results(hardware_results)
        
        if container_results:
            report.container_performance = self._process_container_results(container_results)
        
        if trading_results:
            report.trading_performance = self._process_trading_results(trading_results)
        
        if ai_results:
            report.ai_performance = self._process_ai_results(ai_results)
        
        if stress_results:
            report.stress_test_results = self._process_stress_results(stress_results)
        
        # Generate regression analysis
        report.regression_analysis = self._perform_regression_analysis(report)
        
        # Generate recommendations
        report.recommendations = self._generate_comprehensive_recommendations(report)
        
        # Generate visualizations
        if self.config.include_visualizations and VISUALIZATION_AVAILABLE:
            report.visualizations = await self._generate_visualizations(report)
        
        # Save to historical data
        report_dict = asdict(report)
        self._save_historical_data(report_dict)
        
        # Export in requested formats
        await self._export_report(report)
        
        logger.info(f"Comprehensive report generated: {report_id}")
        return report
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information for the report"""
        try:
            import platform
            import psutil
            
            return {
                "timestamp": datetime.now().isoformat(),
                "platform": platform.platform(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "cpu_cores": os.cpu_count(),
                "total_memory_gb": psutil.virtual_memory().total / (1024**3),
                "architecture": platform.machine()
            }
        except Exception as e:
            logger.warning(f"Could not collect system info: {e}")
            return {"error": str(e)}
    
    def _process_performance_results(self, results: BenchmarkSuiteResult) -> Dict[str, Any]:
        """Process performance benchmark results"""
        
        # Categorize results
        categories = {}
        for result in results.benchmark_results:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(result)
        
        # Calculate category summaries
        category_summaries = {}
        for category, benchmarks in categories.items():
            latencies = [b.duration_ms for b in benchmarks]
            throughputs = [b.throughput for b in benchmarks if b.throughput]
            
            category_summaries[category] = {
                "benchmark_count": len(benchmarks),
                "avg_latency_ms": statistics.mean(latencies),
                "p95_latency_ms": np.percentile(latencies, 95),
                "p99_latency_ms": np.percentile(latencies, 99),
                "avg_throughput": statistics.mean(throughputs) if throughputs else 0,
                "optimization_enabled": all(b.optimization_enabled for b in benchmarks)
            }
        
        # Overall summary
        all_latencies = [b.duration_ms for b in results.benchmark_results]
        all_throughputs = [b.throughput for b in results.benchmark_results if b.throughput]
        
        return {
            "total_benchmarks": len(results.benchmark_results),
            "total_duration_ms": results.total_duration_ms,
            "overall_avg_latency_ms": statistics.mean(all_latencies),
            "overall_p95_latency_ms": np.percentile(all_latencies, 95),
            "overall_p99_latency_ms": np.percentile(all_latencies, 99),
            "avg_throughput": statistics.mean(all_throughputs) if all_throughputs else 0,
            "performance_improvement": results.performance_improvement,
            "regression_status": results.regression_status,
            "category_summaries": category_summaries,
            "hardware_info": results.hardware_info,
            "optimization_summary": results.optimization_summary
        }
    
    def _process_hardware_results(self, results: HardwareValidationReport) -> Dict[str, Any]:
        """Process hardware validation results"""
        
        # Categorize validation results
        categories = {}
        for result in results.validation_results:
            if result.component not in categories:
                categories[result.component] = []
            categories[result.component].append(result)
        
        # Calculate component scores
        component_scores = {}
        for component, validations in categories.items():
            passed = sum(1 for v in validations if v.status.value == "PASS")
            total = len(validations)
            component_scores[component] = (passed / total) * 100 if total > 0 else 0
        
        return {
            "m4_max_detected": results.m4_max_detected,
            "performance_score": results.performance_score,
            "validation_time_ms": results.validation_time_ms,
            "system_info": results.system_info,
            "optimization_compatibility": results.optimization_compatibility,
            "component_scores": component_scores,
            "recommendations": results.recommendations,
            "validation_results_count": len(results.validation_results)
        }
    
    def _process_container_results(self, results: ContainerSuiteResult) -> Dict[str, Any]:
        """Process container benchmark results"""
        
        # Analyze startup times
        startup_results = [r for r in results.container_results if "Startup" in r.test_name]
        startup_times = [r.duration_ms for r in startup_results]
        
        # Analyze resource usage
        resource_results = [r for r in results.container_results if "Resource" in r.test_name]
        memory_usage = [r.memory_usage_mb for r in resource_results]
        cpu_usage = [r.cpu_usage_percent for r in resource_results]
        
        return {
            "total_duration_ms": results.total_duration_ms,
            "containers_tested": results.containers_tested,
            "docker_info": results.docker_info,
            "system_resources": results.system_resources,
            "avg_startup_time_ms": statistics.mean(startup_times) if startup_times else 0,
            "p95_startup_time_ms": np.percentile(startup_times, 95) if startup_times else 0,
            "avg_memory_usage_mb": statistics.mean(memory_usage) if memory_usage else 0,
            "avg_cpu_usage_percent": statistics.mean(cpu_usage) if cpu_usage else 0,
            "performance_improvement": results.performance_improvement,
            "optimization_summary": results.optimization_summary
        }
    
    def _process_trading_results(self, results: TradingBenchmarkSuite) -> Dict[str, Any]:
        """Process trading benchmark results"""
        
        # Categorize trading results
        categories = {}
        for result in results.benchmark_results:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(result)
        
        # Calculate SLA compliance
        sla_compliance_rate = sum(1 for passed in results.sla_compliance.values() if passed) / len(results.sla_compliance) * 100
        
        # Calculate category performance
        category_performance = {}
        for category, benchmarks in categories.items():
            latencies = [b.latency_ms for b in benchmarks]
            throughputs = [b.throughput for b in benchmarks]
            success_rates = [b.success_rate for b in benchmarks]
            
            category_performance[category] = {
                "avg_latency_ms": statistics.mean(latencies),
                "avg_throughput": statistics.mean(throughputs),
                "avg_success_rate": statistics.mean(success_rates),
                "benchmark_count": len(benchmarks)
            }
        
        return {
            "total_duration_ms": results.total_duration_ms,
            "total_benchmarks": len(results.benchmark_results),
            "performance_summary": results.performance_summary,
            "sla_compliance": results.sla_compliance,
            "sla_compliance_rate": sla_compliance_rate,
            "category_performance": category_performance,
            "trading_engine_info": results.trading_engine_info,
            "recommendations": results.recommendations
        }
    
    def _process_ai_results(self, results: AIBenchmarkSuite) -> Dict[str, Any]:
        """Process AI benchmark results"""
        
        # Categorize AI results
        categories = {}
        for result in results.benchmark_results:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(result)
        
        # Calculate category performance
        category_performance = {}
        for category, benchmarks in categories.items():
            inference_times = [b.inference_time_ms for b in benchmarks]
            throughputs = [b.throughput for b in benchmarks]
            accuracies = [b.accuracy for b in benchmarks if b.accuracy is not None]
            neural_usage = [b.neural_engine_usage for b in benchmarks if b.neural_engine_usage is not None]
            
            category_performance[category] = {
                "avg_inference_time_ms": statistics.mean(inference_times),
                "avg_throughput": statistics.mean(throughputs),
                "avg_accuracy": statistics.mean(accuracies) if accuracies else None,
                "avg_neural_engine_usage": statistics.mean(neural_usage) if neural_usage else None,
                "benchmark_count": len(benchmarks)
            }
        
        return {
            "total_duration_ms": results.total_duration_ms,
            "total_benchmarks": len(results.benchmark_results),
            "model_performance_summary": results.model_performance_summary,
            "optimization_effectiveness": results.optimization_effectiveness,
            "neural_engine_info": results.neural_engine_info,
            "category_performance": category_performance,
            "recommendations": results.recommendations
        }
    
    def _process_stress_results(self, results: StressTestSuiteResult) -> Dict[str, Any]:
        """Process stress test results"""
        
        # Categorize stress test results
        categories = {}
        for result in results.stress_test_results:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(result)
        
        # Calculate category reliability
        category_reliability = {}
        for category, tests in categories.items():
            success_count = sum(1 for t in tests if t.success)
            total_count = len(tests)
            
            category_reliability[category] = {
                "success_rate": (success_count / total_count) * 100 if total_count > 0 else 0,
                "avg_peak_cpu": statistics.mean([t.peak_cpu_usage for t in tests]),
                "avg_peak_memory_mb": statistics.mean([t.peak_memory_usage_mb for t in tests]),
                "thermal_issues": sum(1 for t in tests if t.thermal_throttling_detected),
                "memory_leaks": sum(1 for t in tests if t.memory_leaks_detected),
                "test_count": total_count
            }
        
        return {
            "total_duration_ms": results.total_duration_ms,
            "system_stability_score": results.system_stability_score,
            "thermal_performance_score": results.thermal_performance_score,
            "memory_stability_score": results.memory_stability_score,
            "emergency_response_score": results.emergency_response_score,
            "overall_reliability_score": results.overall_reliability_score,
            "category_reliability": category_reliability,
            "critical_issues": results.critical_issues,
            "recommendations": results.recommendations
        }
    
    def _perform_regression_analysis(self, report: BenchmarkReport) -> Dict[str, Any]:
        """Perform regression analysis against historical data"""
        
        if len(self.historical_data) < 2:
            return {"status": "insufficient_data", "message": "Need at least 2 historical data points"}
        
        # Get most recent historical data for comparison
        previous_data = self.historical_data[-1]  # Most recent
        current_scores = self._extract_overall_scores(asdict(report))
        previous_scores = previous_data.get("overall_scores", {})
        
        regressions = []
        improvements = []
        
        for metric, current_value in current_scores.items():
            if metric in previous_scores:
                previous_value = previous_scores[metric]
                
                if previous_value > 0:  # Avoid division by zero
                    change_percent = ((current_value - previous_value) / previous_value) * 100
                    
                    # Determine if this is a regression
                    is_regression = False
                    
                    if "latency" in metric.lower() and change_percent > self.regression_thresholds["latency_increase"]:
                        is_regression = True
                    elif "accuracy" in metric.lower() and change_percent < -self.regression_thresholds["accuracy_drop"]:
                        is_regression = True
                    elif "throughput" in metric.lower() and change_percent < -self.regression_thresholds["throughput_reduction"]:
                        is_regression = True
                    elif "score" in metric.lower() and change_percent < -self.regression_thresholds["performance_degradation"]:
                        is_regression = True
                    
                    change_info = {
                        "metric": metric,
                        "current_value": current_value,
                        "previous_value": previous_value,
                        "change_percent": change_percent,
                        "change_absolute": current_value - previous_value
                    }
                    
                    if is_regression:
                        regressions.append(change_info)
                    elif change_percent > 5:  # Significant improvement
                        improvements.append(change_info)
        
        # Calculate trend analysis
        trend_analysis = self._calculate_trends()
        
        return {
            "status": "completed",
            "comparison_timestamp": previous_data.get("timestamp"),
            "regressions": regressions,
            "improvements": improvements,
            "regression_count": len(regressions),
            "improvement_count": len(improvements),
            "overall_status": "REGRESSION" if regressions else "STABLE" if not improvements else "IMPROVED",
            "trend_analysis": trend_analysis
        }
    
    def _calculate_trends(self) -> Dict[str, Any]:
        """Calculate performance trends over time"""
        
        if len(self.historical_data) < 3:
            return {"status": "insufficient_data"}
        
        # Get last 10 data points for trend analysis
        recent_data = self.historical_data[-10:]
        
        trends = {}
        
        # Analyze trends for key metrics
        key_metrics = ["avg_latency_ms", "avg_throughput", "reliability_score", "hardware_score"]
        
        for metric in key_metrics:
            values = []
            timestamps = []
            
            for entry in recent_data:
                scores = entry.get("overall_scores", {})
                if metric in scores:
                    values.append(scores[metric])
                    timestamps.append(entry.get("timestamp", ""))
            
            if len(values) >= 3:
                # Calculate linear trend
                x = np.arange(len(values))
                z = np.polyfit(x, values, 1)
                slope = z[0]
                
                # Determine trend direction
                if abs(slope) < 0.01:  # Threshold for considering stable
                    trend_direction = "stable"
                elif slope > 0:
                    trend_direction = "improving" if "latency" not in metric else "degrading"
                else:
                    trend_direction = "degrading" if "latency" not in metric else "improving"
                
                trends[metric] = {
                    "direction": trend_direction,
                    "slope": slope,
                    "values": values,
                    "timestamps": timestamps[-5:],  # Last 5 timestamps
                    "latest_value": values[-1],
                    "change_from_oldest": values[-1] - values[0]
                }
        
        return {
            "status": "completed",
            "data_points": len(recent_data),
            "trends": trends
        }
    
    def _generate_comprehensive_recommendations(self, report: BenchmarkReport) -> List[str]:
        """Generate comprehensive recommendations based on all results"""
        
        recommendations = []
        
        # Hardware recommendations
        if report.hardware_validation:
            hw_score = report.hardware_validation.get("performance_score", 0)
            if hw_score < 80:
                recommendations.append("Hardware performance score is below optimal - consider hardware upgrades")
            
            m4_max = report.hardware_validation.get("m4_max_detected", False)
            if not m4_max:
                recommendations.append("M4 Max not detected - ensure running on compatible hardware for optimal performance")
        
        # Performance recommendations
        if report.performance_summary:
            avg_latency = report.performance_summary.get("overall_avg_latency_ms", 0)
            if avg_latency > 50:
                recommendations.append("High average latency detected - investigate performance bottlenecks")
        
        # Trading recommendations
        if report.trading_performance:
            sla_rate = report.trading_performance.get("sla_compliance_rate", 100)
            if sla_rate < 95:
                recommendations.append("SLA compliance below 95% - review and optimize trading performance")
        
        # AI recommendations
        if report.ai_performance:
            neural_usage = report.ai_performance.get("optimization_effectiveness", {}).get("neural_engine_avg_utilization", 0)
            if neural_usage < 60:
                recommendations.append("Neural Engine utilization is low - optimize AI model compilation")
        
        # Stress test recommendations
        if report.stress_test_results:
            reliability = report.stress_test_results.get("overall_reliability_score", 100)
            if reliability < 90:
                recommendations.append("System reliability below 90% - investigate and resolve stability issues")
            
            if report.stress_test_results.get("critical_issues"):
                recommendations.append("Critical issues detected in stress tests - immediate attention required")
        
        # Regression recommendations
        if report.regression_analysis and report.regression_analysis.get("regressions"):
            recommendations.append("Performance regressions detected - investigate recent changes")
        
        # Container recommendations
        if report.container_performance:
            startup_time = report.container_performance.get("avg_startup_time_ms", 0)
            if startup_time > 10000:  # 10 seconds
                recommendations.append("Container startup times are high - optimize container images and configuration")
        
        return recommendations
    
    async def _generate_visualizations(self, report: BenchmarkReport) -> Dict[str, str]:
        """Generate visualizations for the report"""
        
        if not VISUALIZATION_AVAILABLE:
            return {}
        
        visualizations = {}
        
        try:
            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # Performance overview chart
            if report.performance_summary and report.performance_summary.get("category_summaries"):
                viz = await self._create_performance_overview_chart(report.performance_summary["category_summaries"])
                if viz:
                    visualizations["performance_overview"] = viz
            
            # Latency distribution chart
            if report.trading_performance and report.trading_performance.get("category_performance"):
                viz = await self._create_latency_distribution_chart(report.trading_performance["category_performance"])
                if viz:
                    visualizations["latency_distribution"] = viz
            
            # Trend analysis chart
            if report.regression_analysis and report.regression_analysis.get("trend_analysis", {}).get("trends"):
                viz = await self._create_trend_analysis_chart(report.regression_analysis["trend_analysis"]["trends"])
                if viz:
                    visualizations["trend_analysis"] = viz
            
            # System stability radar chart
            if report.stress_test_results:
                viz = await self._create_stability_radar_chart(report.stress_test_results)
                if viz:
                    visualizations["stability_radar"] = viz
            
            # Hardware validation chart
            if report.hardware_validation and report.hardware_validation.get("component_scores"):
                viz = await self._create_hardware_validation_chart(report.hardware_validation["component_scores"])
                if viz:
                    visualizations["hardware_validation"] = viz
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
        
        return visualizations
    
    async def _create_performance_overview_chart(self, category_summaries: Dict[str, Any]) -> Optional[str]:
        """Create performance overview chart"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            categories = list(category_summaries.keys())
            latencies = [category_summaries[cat]["avg_latency_ms"] for cat in categories]
            throughputs = [category_summaries[cat]["avg_throughput"] for cat in categories]
            
            # Latency chart
            ax1.bar(categories, latencies, color='coral')
            ax1.set_title('Average Latency by Category')
            ax1.set_ylabel('Latency (ms)')
            ax1.tick_params(axis='x', rotation=45)
            
            # Throughput chart
            ax2.bar(categories, throughputs, color='skyblue')
            ax2.set_title('Average Throughput by Category')
            ax2.set_ylabel('Throughput (ops/sec)')
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Error creating performance overview chart: {e}")
            return None
    
    async def _create_latency_distribution_chart(self, category_performance: Dict[str, Any]) -> Optional[str]:
        """Create latency distribution chart"""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            categories = list(category_performance.keys())
            latencies = [category_performance[cat]["avg_latency_ms"] for cat in categories]
            
            # Create horizontal bar chart
            y_pos = np.arange(len(categories))
            bars = ax.barh(y_pos, latencies, color=sns.color_palette("viridis", len(categories)))
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(categories)
            ax.set_xlabel('Average Latency (ms)')
            ax.set_title('Trading Performance - Latency Distribution')
            
            # Add value labels on bars
            for i, (bar, latency) in enumerate(zip(bars, latencies)):
                ax.text(bar.get_width() + max(latencies) * 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{latency:.2f}ms', va='center', fontsize=10)
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Error creating latency distribution chart: {e}")
            return None
    
    async def _create_trend_analysis_chart(self, trends: Dict[str, Any]) -> Optional[str]:
        """Create trend analysis chart"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, (metric, trend_data) in enumerate(trends.items()):
                if i >= 4:  # Only show first 4 metrics
                    break
                
                ax = axes[i]
                values = trend_data.get("values", [])
                
                if values:
                    x = range(len(values))
                    ax.plot(x, values, marker='o', linewidth=2, markersize=6)
                    ax.set_title(f'{metric.replace("_", " ").title()}')
                    ax.set_xlabel('Time Period')
                    ax.set_ylabel('Value')
                    ax.grid(True, alpha=0.3)
                    
                    # Add trend line
                    if len(values) >= 2:
                        z = np.polyfit(x, values, 1)
                        p = np.poly1d(z)
                        ax.plot(x, p(x), "--", alpha=0.7, color='red')
                        
                        # Add direction indicator
                        direction = trend_data.get("direction", "stable")
                        color = "green" if direction == "improving" else "red" if direction == "degrading" else "gray"
                        ax.text(0.02, 0.98, f"Trend: {direction}", transform=ax.transAxes, 
                               verticalalignment='top', bbox=dict(boxstyle="round", facecolor=color, alpha=0.3))
            
            # Hide unused subplots
            for i in range(len(trends), 4):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Error creating trend analysis chart: {e}")
            return None
    
    async def _create_stability_radar_chart(self, stress_results: Dict[str, Any]) -> Optional[str]:
        """Create system stability radar chart"""
        try:
            # Stability metrics
            metrics = [
                'System Stability',
                'Thermal Performance',
                'Memory Stability',
                'Emergency Response',
                'Overall Reliability'
            ]
            
            values = [
                stress_results.get("system_stability_score", 0),
                stress_results.get("thermal_performance_score", 0),
                stress_results.get("memory_stability_score", 0),
                stress_results.get("emergency_response_score", 0),
                stress_results.get("overall_reliability_score", 0)
            ]
            
            # Number of variables
            N = len(metrics)
            
            # Compute angles
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Complete the circle
            
            # Add values for completing the circle
            values += values[:1]
            
            # Create radar chart
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            # Plot
            ax.plot(angles, values, 'o-', linewidth=2, label='Current Performance')
            ax.fill(angles, values, alpha=0.25)
            
            # Add labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)
            ax.set_ylim(0, 100)
            
            # Add grid lines at different levels
            ax.set_yticks([20, 40, 60, 80, 100])
            ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
            ax.grid(True)
            
            plt.title('System Stability Radar Chart', size=16, y=1.08)
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Error creating stability radar chart: {e}")
            return None
    
    async def _create_hardware_validation_chart(self, component_scores: Dict[str, float]) -> Optional[str]:
        """Create hardware validation chart"""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            components = list(component_scores.keys())
            scores = list(component_scores.values())
            
            # Color bars based on score
            colors = ['green' if score >= 80 else 'orange' if score >= 60 else 'red' for score in scores]
            
            bars = ax.bar(components, scores, color=colors, alpha=0.7)
            
            # Add score labels on bars
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{score:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            ax.set_ylabel('Validation Score (%)')
            ax.set_title('Hardware Component Validation Scores')
            ax.set_ylim(0, 105)
            
            # Add threshold lines
            ax.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='Good (80%)')
            ax.axhline(y=60, color='orange', linestyle='--', alpha=0.5, label='Fair (60%)')
            
            ax.legend()
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Error creating hardware validation chart: {e}")
            return None
    
    async def _export_report(self, report: BenchmarkReport):
        """Export report in various formats"""
        
        for format_type in self.config.export_formats:
            try:
                if format_type.lower() == "json":
                    await self._export_json(report)
                elif format_type.lower() == "html":
                    await self._export_html(report)
                elif format_type.lower() == "csv":
                    await self._export_csv(report)
                else:
                    logger.warning(f"Unsupported export format: {format_type}")
                    
            except Exception as e:
                logger.error(f"Failed to export report in {format_type} format: {e}")
    
    async def _export_json(self, report: BenchmarkReport):
        """Export report as JSON"""
        filename = os.path.join(self.config.output_directory, f"{report.report_id}.json")
        
        with open(filename, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        
        logger.info(f"Report exported as JSON: {filename}")
    
    async def _export_html(self, report: BenchmarkReport):
        """Export report as HTML"""
        filename = os.path.join(self.config.output_directory, f"{report.report_id}.html")
        
        html_content = self._generate_html_report(report)
        
        with open(filename, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Report exported as HTML: {filename}")
    
    async def _export_csv(self, report: BenchmarkReport):
        """Export report summary as CSV"""
        filename = os.path.join(self.config.output_directory, f"{report.report_id}_summary.csv")
        
        # Create summary data for CSV export
        summary_data = []
        
        # Add performance summary
        if report.performance_summary:
            summary_data.append({
                "Category": "Performance",
                "Metric": "Average Latency",
                "Value": report.performance_summary.get("overall_avg_latency_ms", 0),
                "Unit": "ms"
            })
            summary_data.append({
                "Category": "Performance",
                "Metric": "Average Throughput", 
                "Value": report.performance_summary.get("avg_throughput", 0),
                "Unit": "ops/sec"
            })
        
        # Add hardware validation
        if report.hardware_validation:
            summary_data.append({
                "Category": "Hardware",
                "Metric": "Performance Score",
                "Value": report.hardware_validation.get("performance_score", 0),
                "Unit": "%"
            })
        
        # Add stress test results
        if report.stress_test_results:
            summary_data.append({
                "Category": "Stability",
                "Metric": "Reliability Score",
                "Value": report.stress_test_results.get("overall_reliability_score", 0),
                "Unit": "%"
            })
        
        # Write CSV
        if summary_data:
            fieldnames = ["Category", "Metric", "Value", "Unit"]
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(summary_data)
        
        logger.info(f"Report exported as CSV: {filename}")
    
    def _generate_html_report(self, report: BenchmarkReport) -> str:
        """Generate HTML report content"""
        
        # Start with basic HTML structure
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report.report_id} - M4 Max Benchmark Report</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; }}
                .summary-box {{ background-color: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
                .metric-value {{ font-weight: bold; font-size: 1.2em; color: #2980b9; }}
                .status-pass {{ color: #27ae60; font-weight: bold; }}
                .status-fail {{ color: #e74c3c; font-weight: bold; }}
                .status-warning {{ color: #f39c12; font-weight: bold; }}
                .recommendation {{ background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; margin: 5px 0; border-radius: 5px; }}
                .visualization {{ text-align: center; margin: 20px 0; }}
                .visualization img {{ max-width: 100%; border-radius: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                .table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                .table th, .table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .table th {{ background-color: #3498db; color: white; }}
                .table tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .regression {{ background-color: #ffebee; border: 1px solid #f8bbd9; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .improvement {{ background-color: #e8f5e8; border: 1px solid #c8e6c9; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            </style>
        </head>
        <body>
        <div class="container">
        """
        
        # Report header
        html += f"""
        <h1>M4 Max Benchmark Report</h1>
        <div class="summary-box">
            <h3>Report Information</h3>
            <div class="metric">Report ID: <span class="metric-value">{report.report_id}</span></div>
            <div class="metric">Generated: <span class="metric-value">{report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</span></div>
            <div class="metric">Platform: <span class="metric-value">{report.system_info.get('platform', 'Unknown')}</span></div>
            <div class="metric">Architecture: <span class="metric-value">{report.system_info.get('architecture', 'Unknown')}</span></div>
        </div>
        """
        
        # Executive Summary
        html += self._generate_executive_summary_html(report)
        
        # Performance Results
        if report.performance_summary:
            html += self._generate_performance_section_html(report.performance_summary)
        
        # Hardware Validation
        if report.hardware_validation:
            html += self._generate_hardware_section_html(report.hardware_validation)
        
        # Trading Performance
        if report.trading_performance:
            html += self._generate_trading_section_html(report.trading_performance)
        
        # AI Performance
        if report.ai_performance:
            html += self._generate_ai_section_html(report.ai_performance)
        
        # Stress Test Results
        if report.stress_test_results:
            html += self._generate_stress_section_html(report.stress_test_results)
        
        # Regression Analysis
        if report.regression_analysis:
            html += self._generate_regression_section_html(report.regression_analysis)
        
        # Visualizations
        if report.visualizations:
            html += self._generate_visualizations_section_html(report.visualizations)
        
        # Recommendations
        if report.recommendations:
            html += f"""
            <h2>Recommendations</h2>
            """
            for rec in report.recommendations:
                html += f'<div class="recommendation">{rec}</div>'
        
        # Close HTML
        html += """
        </div>
        </body>
        </html>
        """
        
        return html
    
    def _generate_executive_summary_html(self, report: BenchmarkReport) -> str:
        """Generate executive summary HTML section"""
        
        html = """<h2>Executive Summary</h2><div class="summary-box">"""
        
        # Overall system status
        overall_status = "HEALTHY"
        status_class = "status-pass"
        
        # Check for critical issues
        if report.stress_test_results and report.stress_test_results.get("critical_issues"):
            overall_status = "CRITICAL ISSUES"
            status_class = "status-fail"
        elif report.regression_analysis and report.regression_analysis.get("regressions"):
            overall_status = "REGRESSIONS DETECTED"
            status_class = "status-warning"
        
        html += f'<div class="metric">System Status: <span class="{status_class}">{overall_status}</span></div>'
        
        # Key metrics
        if report.performance_summary:
            html += f'<div class="metric">Average Latency: <span class="metric-value">{report.performance_summary.get("overall_avg_latency_ms", 0):.2f}ms</span></div>'
        
        if report.hardware_validation:
            html += f'<div class="metric">Hardware Score: <span class="metric-value">{report.hardware_validation.get("performance_score", 0):.1f}%</span></div>'
        
        if report.stress_test_results:
            html += f'<div class="metric">Reliability Score: <span class="metric-value">{report.stress_test_results.get("overall_reliability_score", 0):.1f}%</span></div>'
        
        html += "</div>"
        return html
    
    def _generate_performance_section_html(self, performance_data: Dict[str, Any]) -> str:
        """Generate performance section HTML"""
        
        html = f"""
        <h2>Performance Benchmarks</h2>
        <div class="summary-box">
            <div class="metric">Total Benchmarks: <span class="metric-value">{performance_data.get('total_benchmarks', 0)}</span></div>
            <div class="metric">Average Latency: <span class="metric-value">{performance_data.get('overall_avg_latency_ms', 0):.2f}ms</span></div>
            <div class="metric">P95 Latency: <span class="metric-value">{performance_data.get('overall_p95_latency_ms', 0):.2f}ms</span></div>
            <div class="metric">Average Throughput: <span class="metric-value">{performance_data.get('avg_throughput', 0):.0f} ops/sec</span></div>
            <div class="metric">Regression Status: <span class="{'status-pass' if performance_data.get('regression_status') == 'PASS' else 'status-fail'}">{performance_data.get('regression_status', 'UNKNOWN')}</span></div>
        </div>
        """
        
        # Category breakdown
        if performance_data.get("category_summaries"):
            html += "<h3>Performance by Category</h3><table class='table'>"
            html += "<tr><th>Category</th><th>Avg Latency (ms)</th><th>P95 Latency (ms)</th><th>Avg Throughput</th><th>Optimized</th></tr>"
            
            for category, data in performance_data["category_summaries"].items():
                optimized = "✓" if data.get("optimization_enabled") else "✗"
                html += f"""
                <tr>
                    <td>{category}</td>
                    <td>{data.get('avg_latency_ms', 0):.2f}</td>
                    <td>{data.get('p95_latency_ms', 0):.2f}</td>
                    <td>{data.get('avg_throughput', 0):.0f}</td>
                    <td>{optimized}</td>
                </tr>
                """
            html += "</table>"
        
        return html
    
    def _generate_hardware_section_html(self, hardware_data: Dict[str, Any]) -> str:
        """Generate hardware validation section HTML"""
        
        html = f"""
        <h2>Hardware Validation</h2>
        <div class="summary-box">
            <div class="metric">M4 Max Detected: <span class="{'status-pass' if hardware_data.get('m4_max_detected') else 'status-fail'}">{'Yes' if hardware_data.get('m4_max_detected') else 'No'}</span></div>
            <div class="metric">Performance Score: <span class="metric-value">{hardware_data.get('performance_score', 0):.1f}%</span></div>
            <div class="metric">Validation Time: <span class="metric-value">{hardware_data.get('validation_time_ms', 0):.2f}ms</span></div>
        </div>
        """
        
        # Component scores
        if hardware_data.get("component_scores"):
            html += "<h3>Component Validation Scores</h3><table class='table'>"
            html += "<tr><th>Component</th><th>Score (%)</th><th>Status</th></tr>"
            
            for component, score in hardware_data["component_scores"].items():
                if score >= 80:
                    status = '<span class="status-pass">PASS</span>'
                elif score >= 60:
                    status = '<span class="status-warning">WARNING</span>'
                else:
                    status = '<span class="status-fail">FAIL</span>'
                
                html += f"""
                <tr>
                    <td>{component}</td>
                    <td>{score:.1f}</td>
                    <td>{status}</td>
                </tr>
                """
            html += "</table>"
        
        return html
    
    def _generate_trading_section_html(self, trading_data: Dict[str, Any]) -> str:
        """Generate trading performance section HTML"""
        
        html = f"""
        <h2>Trading Performance</h2>
        <div class="summary-box">
            <div class="metric">Total Benchmarks: <span class="metric-value">{trading_data.get('total_benchmarks', 0)}</span></div>
            <div class="metric">SLA Compliance: <span class="metric-value">{trading_data.get('sla_compliance_rate', 0):.1f}%</span></div>
        </div>
        """
        
        return html
    
    def _generate_ai_section_html(self, ai_data: Dict[str, Any]) -> str:
        """Generate AI performance section HTML"""
        
        html = f"""
        <h2>AI/ML Performance</h2>
        <div class="summary-box">
            <div class="metric">Total Benchmarks: <span class="metric-value">{ai_data.get('total_benchmarks', 0)}</span></div>
        </div>
        """
        
        return html
    
    def _generate_stress_section_html(self, stress_data: Dict[str, Any]) -> str:
        """Generate stress test section HTML"""
        
        html = f"""
        <h2>Stress Test Results</h2>
        <div class="summary-box">
            <div class="metric">System Stability: <span class="metric-value">{stress_data.get('system_stability_score', 0):.1f}%</span></div>
            <div class="metric">Thermal Performance: <span class="metric-value">{stress_data.get('thermal_performance_score', 0):.1f}%</span></div>
            <div class="metric">Memory Stability: <span class="metric-value">{stress_data.get('memory_stability_score', 0):.1f}%</span></div>
            <div class="metric">Overall Reliability: <span class="metric-value">{stress_data.get('overall_reliability_score', 0):.1f}%</span></div>
        </div>
        """
        
        # Critical issues
        if stress_data.get("critical_issues"):
            html += "<h3>Critical Issues</h3>"
            for issue in stress_data["critical_issues"]:
                html += f'<div class="regression">{issue}</div>'
        
        return html
    
    def _generate_regression_section_html(self, regression_data: Dict[str, Any]) -> str:
        """Generate regression analysis section HTML"""
        
        html = f"""
        <h2>Regression Analysis</h2>
        <div class="summary-box">
            <div class="metric">Status: <span class="{'status-pass' if regression_data.get('overall_status') in ['STABLE', 'IMPROVED'] else 'status-fail'}">{regression_data.get('overall_status', 'UNKNOWN')}</span></div>
            <div class="metric">Regressions: <span class="metric-value">{regression_data.get('regression_count', 0)}</span></div>
            <div class="metric">Improvements: <span class="metric-value">{regression_data.get('improvement_count', 0)}</span></div>
        </div>
        """
        
        # Regressions
        if regression_data.get("regressions"):
            html += "<h3>Performance Regressions</h3>"
            for regression in regression_data["regressions"]:
                html += f"""
                <div class="regression">
                    <strong>{regression['metric']}</strong>: 
                    {regression['change_percent']:+.1f}% change 
                    ({regression['previous_value']:.2f} → {regression['current_value']:.2f})
                </div>
                """
        
        # Improvements  
        if regression_data.get("improvements"):
            html += "<h3>Performance Improvements</h3>"
            for improvement in regression_data["improvements"]:
                html += f"""
                <div class="improvement">
                    <strong>{improvement['metric']}</strong>: 
                    {improvement['change_percent']:+.1f}% improvement 
                    ({improvement['previous_value']:.2f} → {improvement['current_value']:.2f})
                </div>
                """
        
        return html
    
    def _generate_visualizations_section_html(self, visualizations: Dict[str, str]) -> str:
        """Generate visualizations section HTML"""
        
        html = "<h2>Performance Visualizations</h2>"
        
        for viz_name, viz_data in visualizations.items():
            title = viz_name.replace('_', ' ').title()
            html += f"""
            <h3>{title}</h3>
            <div class="visualization">
                <img src="data:image/png;base64,{viz_data}" alt="{title}">
            </div>
            """
        
        return html
    
    def _get_html_template(self) -> str:
        """Get HTML template for reports"""
        # This would contain a more sophisticated HTML template
        # For now, we generate HTML programmatically
        return ""
    
    def get_historical_summary(self) -> Dict[str, Any]:
        """Get summary of historical benchmark data"""
        
        if not self.historical_data:
            return {"status": "no_data", "message": "No historical data available"}
        
        # Calculate trends over time
        recent_entries = self.historical_data[-10:]  # Last 10 entries
        
        summary = {
            "total_reports": len(self.historical_data),
            "date_range": {
                "first": self.historical_data[0].get("timestamp"),
                "last": self.historical_data[-1].get("timestamp")
            },
            "recent_trends": {}
        }
        
        # Analyze trends for key metrics
        key_metrics = ["avg_latency_ms", "reliability_score", "hardware_score"]
        
        for metric in key_metrics:
            values = []
            for entry in recent_entries:
                scores = entry.get("overall_scores", {})
                if metric in scores:
                    values.append(scores[metric])
            
            if len(values) >= 2:
                trend = "improving" if values[-1] > values[0] else "declining" if values[-1] < values[0] else "stable"
                change = ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
                
                summary["recent_trends"][metric] = {
                    "trend": trend,
                    "change_percent": change,
                    "latest_value": values[-1],
                    "data_points": len(values)
                }
        
        return summary

# Convenience function for quick report generation
async def generate_quick_report(
    performance_results: Optional[BenchmarkSuiteResult] = None,
    **kwargs
) -> BenchmarkReport:
    """Generate a quick benchmark report with default configuration"""
    
    reporter = BenchmarkReporter()
    return await reporter.generate_comprehensive_report(
        performance_results=performance_results,
        **kwargs
    )