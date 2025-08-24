#!/usr/bin/env python3
"""
M4 Max Benchmarking Suite Runner
===============================

Main entry point for running comprehensive M4 Max benchmarking and validation.

Usage:
    python run_benchmarks.py [options]

Options:
    --suite [all|performance|hardware|containers|trading|ai|stress]
    --output-dir PATH
    --format [json|html|csv]
    --quick-run (reduced iterations for faster execution)
    --regression-check (only run regression tests)
    --export-baselines (save current results as new baselines)

Examples:
    # Run full benchmark suite
    python run_benchmarks.py --suite all --format html

    # Run only trading benchmarks
    python run_benchmarks.py --suite trading --output-dir ./results

    # Quick performance check
    python run_benchmarks.py --suite performance --quick-run

    # Regression testing
    python run_benchmarks.py --regression-check
"""

import asyncio
import argparse
import logging
import sys
import os
import time
from datetime import datetime
from typing import Optional, List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('benchmark_run.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Import benchmark suites
from .performance_suite import PerformanceBenchmarkSuite
from .hardware_validation import HardwareValidator
from .container_benchmarks import ContainerBenchmarks
from .trading_benchmarks import TradingBenchmarks
from .ai_benchmarks import AIBenchmarks
from .stress_tests import StressTestSuite
from .benchmark_reporter import BenchmarkReporter, ReportConfig

class BenchmarkRunner:
    """
    Main benchmark runner orchestrating all benchmark suites
    """
    
    def __init__(self, args):
        self.args = args
        
        # Configure output directory
        self.output_dir = args.output_dir or "benchmark_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Configure report settings
        self.report_config = ReportConfig(
            include_visualizations=not args.quick_run,
            include_raw_data=True,
            include_recommendations=True,
            include_historical_comparison=True,
            export_formats=args.format if isinstance(args.format, list) else [args.format],
            output_directory=self.output_dir
        )
        
        # Initialize benchmark suites
        self._init_benchmark_suites()
        
        # Initialize reporter
        self.reporter = BenchmarkReporter(self.report_config)
    
    def _init_benchmark_suites(self):
        """Initialize all benchmark suites with appropriate configuration"""
        
        # Base configuration for quick runs
        quick_config = {
            "iterations": 50 if self.args.quick_run else 100,
            "warmup_iterations": 5 if self.args.quick_run else 10,
        } if hasattr(self.args, 'quick_run') and self.args.quick_run else {}
        
        # Initialize suites
        self.performance_suite = PerformanceBenchmarkSuite(quick_config)
        self.hardware_validator = HardwareValidator()
        self.container_benchmarks = ContainerBenchmarks(quick_config)
        self.trading_benchmarks = TradingBenchmarks(quick_config)
        self.ai_benchmarks = AIBenchmarks(quick_config)
        self.stress_tests = StressTestSuite({
            **quick_config,
            "stress_duration": 60 if self.args.quick_run else 300,  # 1 min vs 5 min
            "extreme_stress_duration": 120 if self.args.quick_run else 600  # 2 min vs 10 min
        })
    
    async def run_benchmarks(self) -> bool:
        """
        Run the requested benchmark suites
        """
        logger.info("=" * 60)
        logger.info("M4 Max Benchmarking Suite Started")
        logger.info("=" * 60)
        
        start_time = time.time()
        results = {}
        
        try:
            # Determine which suites to run
            suites_to_run = self._get_suites_to_run()
            
            logger.info(f"Running benchmark suites: {', '.join(suites_to_run)}")
            
            # Run each suite
            if "hardware" in suites_to_run:
                logger.info("Running Hardware Validation...")
                results["hardware"] = await self.hardware_validator.validate_hardware()
                logger.info("✓ Hardware Validation completed")
            
            if "performance" in suites_to_run:
                logger.info("Running Performance Benchmarks...")
                results["performance"] = await self.performance_suite.run_full_benchmark()
                logger.info("✓ Performance Benchmarks completed")
            
            if "containers" in suites_to_run:
                logger.info("Running Container Benchmarks...")
                try:
                    results["containers"] = await self.container_benchmarks.run_container_benchmarks()
                    logger.info("✓ Container Benchmarks completed")
                except Exception as e:
                    logger.warning(f"Container benchmarks failed (Docker may not be available): {e}")
                    results["containers"] = None
            
            if "trading" in suites_to_run:
                logger.info("Running Trading Benchmarks...")
                results["trading"] = await self.trading_benchmarks.run_trading_benchmarks()
                logger.info("✓ Trading Benchmarks completed")
            
            if "ai" in suites_to_run:
                logger.info("Running AI/ML Benchmarks...")
                results["ai"] = await self.ai_benchmarks.run_ai_benchmarks()
                logger.info("✓ AI/ML Benchmarks completed")
            
            if "stress" in suites_to_run:
                logger.info("Running Stress Tests...")
                results["stress"] = await self.stress_tests.run_stress_tests()
                logger.info("✓ Stress Tests completed")
            
            # Generate comprehensive report
            logger.info("Generating comprehensive report...")
            report = await self.reporter.generate_comprehensive_report(
                performance_results=results.get("performance"),
                hardware_results=results.get("hardware"),
                container_results=results.get("containers"),
                trading_results=results.get("trading"),
                ai_results=results.get("ai"),
                stress_results=results.get("stress")
            )
            
            # Export baselines if requested
            if hasattr(self.args, 'export_baselines') and self.args.export_baselines:
                if results.get("performance"):
                    self.performance_suite.save_baselines()
                    logger.info("✓ New baselines exported")
            
            # Print summary
            self._print_benchmark_summary(results, report, time.time() - start_time)
            
            return True
            
        except Exception as e:
            logger.error(f"Benchmark execution failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _get_suites_to_run(self) -> List[str]:
        """Determine which benchmark suites to run"""
        
        if hasattr(self.args, 'regression_check') and self.args.regression_check:
            # For regression checks, run key performance tests
            return ["performance", "trading"]
        
        suite_arg = getattr(self.args, 'suite', 'all')
        
        if suite_arg == "all":
            return ["hardware", "performance", "containers", "trading", "ai", "stress"]
        elif suite_arg == "performance":
            return ["hardware", "performance"]
        elif suite_arg == "hardware":
            return ["hardware"]
        elif suite_arg == "containers":
            return ["containers"]
        elif suite_arg == "trading":
            return ["trading"]
        elif suite_arg == "ai":
            return ["ai"]
        elif suite_arg == "stress":
            return ["stress"]
        else:
            logger.warning(f"Unknown suite: {suite_arg}, running all suites")
            return ["hardware", "performance", "containers", "trading", "ai", "stress"]
    
    def _print_benchmark_summary(self, results, report, execution_time):
        """Print a summary of benchmark results"""
        
        logger.info("=" * 60)
        logger.info("BENCHMARK EXECUTION SUMMARY")
        logger.info("=" * 60)
        
        logger.info(f"Total Execution Time: {execution_time:.2f} seconds")
        logger.info(f"Report ID: {report.report_id}")
        logger.info(f"Output Directory: {self.output_dir}")
        
        # Hardware summary
        if results.get("hardware"):
            hw_result = results["hardware"]
            logger.info(f"Hardware Validation Score: {hw_result.performance_score:.1f}%")
            logger.info(f"M4 Max Detected: {'Yes' if hw_result.m4_max_detected else 'No'}")
        
        # Performance summary
        if results.get("performance"):
            perf_result = results["performance"]
            logger.info(f"Average Latency: {report.performance_summary.get('overall_avg_latency_ms', 0):.2f}ms")
            logger.info(f"Performance Regression Status: {perf_result.regression_status}")
        
        # Trading summary
        if results.get("trading"):
            trading_result = results["trading"]
            sla_rate = report.trading_performance.get('sla_compliance_rate', 0)
            logger.info(f"Trading SLA Compliance: {sla_rate:.1f}%")
        
        # Stress test summary
        if results.get("stress"):
            stress_result = results["stress"]
            logger.info(f"System Reliability Score: {stress_result.overall_reliability_score:.1f}%")
            if stress_result.critical_issues:
                logger.warning(f"Critical Issues Detected: {len(stress_result.critical_issues)}")
        
        # Recommendations
        if report.recommendations:
            logger.info("KEY RECOMMENDATIONS:")
            for i, rec in enumerate(report.recommendations[:5], 1):  # Show top 5
                logger.info(f"  {i}. {rec}")
        
        # Report files generated
        logger.info("GENERATED REPORTS:")
        for format_type in self.report_config.export_formats:
            filename = f"{report.report_id}.{format_type}"
            filepath = os.path.join(self.output_dir, filename)
            if os.path.exists(filepath):
                logger.info(f"  {format_type.upper()}: {filepath}")
        
        logger.info("=" * 60)
        logger.info("Benchmark execution completed successfully!")
        logger.info("=" * 60)

def create_parser():
    """Create command line argument parser"""
    
    parser = argparse.ArgumentParser(
        description="M4 Max Comprehensive Benchmarking Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --suite all --format html
  %(prog)s --suite trading --quick-run
  %(prog)s --regression-check
  %(prog)s --suite performance --export-baselines
        """
    )
    
    parser.add_argument(
        "--suite",
        choices=["all", "performance", "hardware", "containers", "trading", "ai", "stress"],
        default="all",
        help="Benchmark suite to run (default: all)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Output directory for results (default: benchmark_results)"
    )
    
    parser.add_argument(
        "--format",
        nargs="+",
        choices=["json", "html", "csv"],
        default=["html", "json"],
        help="Output format(s) (default: html json)"
    )
    
    parser.add_argument(
        "--quick-run",
        action="store_true",
        help="Run with reduced iterations for faster execution"
    )
    
    parser.add_argument(
        "--regression-check",
        action="store_true",
        help="Run only regression tests against baselines"
    )
    
    parser.add_argument(
        "--export-baselines",
        action="store_true",
        help="Save current results as new performance baselines"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        default="benchmark_run.log",
        help="Log file path (default: benchmark_run.log)"
    )
    
    return parser

async def main():
    """Main entry point"""
    
    parser = create_parser()
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Update log file if specified
    if args.log_file != "benchmark_run.log":
        # Remove existing file handlers and add new one
        logger = logging.getLogger()
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)
        logger.addHandler(logging.FileHandler(args.log_file, mode='a'))
    
    # Create and run benchmark runner
    runner = BenchmarkRunner(args)
    
    try:
        success = await runner.run_benchmarks()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("Benchmark execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Handle event loop for different Python versions
    try:
        # Python 3.7+
        asyncio.run(main())
    except AttributeError:
        # Python 3.6
        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(main())
        finally:
            loop.close()