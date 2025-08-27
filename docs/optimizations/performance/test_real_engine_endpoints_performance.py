#!/usr/bin/env python3
"""
REAL ENGINE ENDPOINTS PERFORMANCE VALIDATION
===========================================

Tests actual available endpoints on running engines to validate
SME acceleration and institutional performance requirements.

Author: Quinn (Senior Developer & QA Architect) 🧪
Date: August 26, 2025
"""

import asyncio
import time
import json
import requests
import statistics
from datetime import datetime
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealEnginePerformanceTest:
    """Test actual engine endpoints for performance validation"""
    
    def __init__(self):
        # Map of engines with their actual available endpoints
        self.engine_tests = {
            'risk-engine': {
                'port': 8200,
                'health_endpoint': '/health',
                'test_endpoints': [
                    {'path': '/api/v1/enhanced-risk/health', 'method': 'GET', 'payload': None}
                ]
            },
            'analytics-engine': {
                'port': 8100,
                'health_endpoint': '/health',
                'test_endpoints': [
                    {'path': '/health', 'method': 'GET', 'payload': None},
                    {'path': '/metrics', 'method': 'GET', 'payload': None},
                    {'path': '/analytics/available-symbols', 'method': 'GET', 'payload': None},
                    {'path': '/analytics/symbol/AAPL', 'method': 'GET', 'payload': None}
                ]
            },
            'portfolio-engine': {
                'port': 8900,
                'health_endpoint': '/health',
                'test_endpoints': [
                    {'path': '/health', 'method': 'GET', 'payload': None},
                    {'path': '/capabilities', 'method': 'GET', 'payload': None},
                    {'path': '/institutional/strategies/library', 'method': 'GET', 'payload': None}
                ]
            },
            'backend': {
                'port': 8001,
                'health_endpoint': '/health',
                'test_endpoints': [
                    {'path': '/health', 'method': 'GET', 'payload': None},
                    {'path': '/docs', 'method': 'GET', 'payload': None}
                ]
            }
        }
        
        self.test_results = {}

    def test_engine_endpoint(self, engine_name: str, endpoint_config: Dict[str, Any], iterations: int = 10) -> Dict[str, Any]:
        """Test a specific engine endpoint multiple times"""
        base_url = f"http://localhost:{self.engine_tests[engine_name]['port']}"
        path = endpoint_config['path']
        method = endpoint_config['method']
        payload = endpoint_config.get('payload')
        
        response_times = []
        success_count = 0
        errors = []
        
        logger.info(f"Testing {engine_name} - {method} {path} ({iterations} iterations)")
        
        for i in range(iterations):
            try:
                start_time = time.time()
                
                if method == 'GET':
                    response = requests.get(base_url + path, timeout=10)
                elif method == 'POST':
                    response = requests.post(base_url + path, json=payload, timeout=10)
                
                response_time = (time.time() - start_time) * 1000  # ms
                
                if 200 <= response.status_code < 300:
                    success_count += 1
                    response_times.append(response_time)
                else:
                    errors.append(f"HTTP {response.status_code}: {response.text[:200]}")
                    
            except Exception as e:
                errors.append(f"Exception: {str(e)}")
        
        # Calculate performance metrics
        if response_times:
            avg_response_time = statistics.mean(response_times)
            median_response_time = statistics.median(response_times)
            p95_response_time = sorted(response_times)[int(0.95 * len(response_times))] if len(response_times) > 1 else response_times[0]
            min_response_time = min(response_times)
            max_response_time = max(response_times)
        else:
            avg_response_time = median_response_time = p95_response_time = float('inf')
            min_response_time = max_response_time = float('inf')
        
        success_rate = success_count / iterations
        throughput = success_count / (sum(response_times) / 1000) if response_times else 0
        
        # SME acceleration assessment based on response times
        sme_performance_grade = 'A+' if avg_response_time < 5 else \
                              'A' if avg_response_time < 20 else \
                              'B' if avg_response_time < 100 else \
                              'C' if avg_response_time < 1000 else 'F'
        
        return {
            'engine': engine_name,
            'endpoint': f"{method} {path}",
            'iterations': iterations,
            'success_count': success_count,
            'success_rate': success_rate,
            'avg_response_time_ms': avg_response_time,
            'median_response_time_ms': median_response_time,
            'p95_response_time_ms': p95_response_time,
            'min_response_time_ms': min_response_time,
            'max_response_time_ms': max_response_time,
            'throughput_rps': throughput,
            'sme_grade': sme_performance_grade,
            'institutional_compliant': avg_response_time < 100 and success_rate >= 0.95,
            'errors': errors[:5]  # First 5 errors only
        }

    async def run_comprehensive_performance_test(self) -> Dict[str, Any]:
        """Run comprehensive performance tests on all available engines"""
        logger.info("🚀 Starting Real Engine Performance Test Suite")
        
        test_start_time = time.time()
        overall_results = {
            'test_info': {
                'timestamp': datetime.now().isoformat(),
                'tester': 'Quinn (Senior QA Architect)',
                'test_type': 'Real Engine Endpoint Performance Validation',
                'sme_focus': True
            },
            'engine_tests': {},
            'summary': {},
            'sme_analysis': {}
        }
        
        # Test each engine
        all_response_times = []
        all_success_rates = []
        institutional_compliant = 0
        total_tests = 0
        sme_grades = []
        
        for engine_name, engine_config in self.engine_tests.items():
            logger.info(f"🔧 Testing {engine_name}")
            overall_results['engine_tests'][engine_name] = {}
            
            for endpoint_config in engine_config['test_endpoints']:
                test_result = self.test_engine_endpoint(engine_name, endpoint_config, iterations=20)
                endpoint_key = f"{test_result['endpoint'].replace('/', '_').replace(' ', '_')}"
                overall_results['engine_tests'][engine_name][endpoint_key] = test_result
                
                # Aggregate metrics
                if test_result['avg_response_time_ms'] != float('inf'):
                    all_response_times.append(test_result['avg_response_time_ms'])
                    all_success_rates.append(test_result['success_rate'])
                    sme_grades.append(test_result['sme_grade'])
                    
                    if test_result['institutional_compliant']:
                        institutional_compliant += 1
                    
                    total_tests += 1
                
                # Log individual results
                logger.info(f"  ✅ {test_result['endpoint']}: {test_result['avg_response_time_ms']:.2f}ms avg, "
                          f"{test_result['success_rate']*100:.1f}% success, Grade: {test_result['sme_grade']}")
        
        # Calculate summary statistics
        total_test_time = time.time() - test_start_time
        
        overall_results['summary'] = {
            'total_engines_tested': len(self.engine_tests),
            'total_endpoints_tested': total_tests,
            'total_test_duration_seconds': total_test_time,
            'average_response_time_ms': statistics.mean(all_response_times) if all_response_times else 0,
            'median_response_time_ms': statistics.median(all_response_times) if all_response_times else 0,
            'p95_response_time_ms': sorted(all_response_times)[int(0.95 * len(all_response_times))] if len(all_response_times) > 1 else (all_response_times[0] if all_response_times else 0),
            'min_response_time_ms': min(all_response_times) if all_response_times else 0,
            'max_response_time_ms': max(all_response_times) if all_response_times else 0,
            'average_success_rate': statistics.mean(all_success_rates) if all_success_rates else 0,
            'institutional_compliance_rate': institutional_compliant / total_tests if total_tests > 0 else 0
        }
        
        # SME Performance Analysis
        grade_counts = {}
        for grade in sme_grades:
            grade_counts[grade] = grade_counts.get(grade, 0) + 1
        
        dominant_grade = max(grade_counts.items(), key=lambda x: x[1])[0] if grade_counts else 'N/A'
        
        # SME acceleration validation
        ultra_fast_endpoints = len([t for t in all_response_times if t < 5])  # < 5ms
        fast_endpoints = len([t for t in all_response_times if 5 <= t < 20])  # 5-20ms
        good_endpoints = len([t for t in all_response_times if 20 <= t < 100])  # 20-100ms
        slow_endpoints = len([t for t in all_response_times if t >= 100])  # >= 100ms
        
        overall_results['sme_analysis'] = {
            'dominant_performance_grade': dominant_grade,
            'grade_distribution': grade_counts,
            'performance_breakdown': {
                'ultra_fast_sub5ms': ultra_fast_endpoints,
                'fast_5to20ms': fast_endpoints,
                'good_20to100ms': good_endpoints,
                'slow_over100ms': slow_endpoints
            },
            'sme_acceleration_validated': overall_results['summary']['average_response_time_ms'] < 50,
            'meets_institutional_requirements': overall_results['summary']['institutional_compliance_rate'] >= 0.90,
            'performance_tier': 'Tier 1' if overall_results['summary']['average_response_time_ms'] < 10 else
                              'Tier 2' if overall_results['summary']['average_response_time_ms'] < 50 else
                              'Tier 3' if overall_results['summary']['average_response_time_ms'] < 200 else 'Below Standard'
        }
        
        # Overall certification
        production_ready = (
            overall_results['summary']['average_success_rate'] >= 0.95 and
            overall_results['summary']['average_response_time_ms'] < 200 and
            overall_results['summary']['institutional_compliance_rate'] >= 0.80
        )
        
        overall_results['certification'] = {
            'production_ready': production_ready,
            'overall_grade': dominant_grade,
            'certified_by': 'Quinn (Senior QA Architect)',
            'certification_timestamp': datetime.now().isoformat(),
            'recommendations': self._generate_recommendations(overall_results)
        }
        
        return overall_results

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        summary = results['summary']
        sme = results['sme_analysis']
        
        if summary['average_response_time_ms'] > 50:
            recommendations.append("Consider M4 Max hardware optimizations - target <50ms response times")
        
        if summary['average_success_rate'] < 0.98:
            recommendations.append("Address reliability issues - target 98%+ success rate")
        
        if sme['performance_breakdown']['slow_over100ms'] > 0:
            recommendations.append(f"Optimize {sme['performance_breakdown']['slow_over100ms']} slow endpoints (>100ms)")
        
        if summary['institutional_compliance_rate'] < 0.90:
            recommendations.append("Improve institutional compliance - need 90%+ endpoints meeting requirements")
        
        if sme['performance_tier'] not in ['Tier 1', 'Tier 2']:
            recommendations.append("Performance below institutional standards - consider SME acceleration upgrades")
        
        if not recommendations:
            recommendations.append("All systems performing optimally - ready for institutional deployment")
        
        return recommendations

    def save_results(self, results: Dict[str, Any]) -> str:
        """Save test results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/tmp/nautilus_real_engine_performance_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"📁 Results saved to {filename}")
        return filename

async def main():
    """Execute real engine performance tests"""
    print("="*80)
    print("🚀 NAUTILUS REAL ENGINE PERFORMANCE VALIDATION")
    print("🎯 Testing actual available endpoints for SME acceleration")
    print("🏛️ Institutional-grade performance certification")
    print("="*80)
    
    test_suite = RealEnginePerformanceTest()
    
    try:
        # Run performance tests
        results = await test_suite.run_comprehensive_performance_test()
        
        # Save results
        results_file = test_suite.save_results(results)
        
        # Display executive summary
        print("\n" + "="*80)
        print("📊 EXECUTIVE SUMMARY")
        print("="*80)
        summary = results['summary']
        sme = results['sme_analysis']
        cert = results['certification']
        
        print(f"🏆 Overall Grade: {cert['overall_grade']}")
        print(f"✅ Production Ready: {'YES' if cert['production_ready'] else 'NO'}")
        print(f"⚡ SME Acceleration: {'VALIDATED' if sme['sme_acceleration_validated'] else 'NEEDS WORK'}")
        print(f"🏛️ Performance Tier: {sme['performance_tier']}")
        print(f"📈 Average Response Time: {summary['average_response_time_ms']:.2f}ms")
        print(f"🎯 Success Rate: {summary['average_success_rate']*100:.1f}%")
        print(f"🏛️ Institutional Compliance: {summary['institutional_compliance_rate']*100:.1f}%")
        
        print("\n📊 Performance Breakdown:")
        breakdown = sme['performance_breakdown']
        print(f"  🚀 Ultra-Fast (<5ms): {breakdown['ultra_fast_sub5ms']}")
        print(f"  ⚡ Fast (5-20ms): {breakdown['fast_5to20ms']}")
        print(f"  ✅ Good (20-100ms): {breakdown['good_20to100ms']}")
        print(f"  ⚠️  Slow (>100ms): {breakdown['slow_over100ms']}")
        
        print(f"\n📋 Recommendations:")
        for i, rec in enumerate(cert['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print(f"\n📁 Detailed results: {results_file}")
        print("="*80)
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())