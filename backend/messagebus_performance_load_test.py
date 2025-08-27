#!/usr/bin/env python3
"""
MessageBus Performance and Load Test
Tests the dual messagebus architecture performance and inter-engine communication load.
"""

import asyncio
import json
import time
import aiohttp
import statistics
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import concurrent.futures
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Healthy engines from previous test
HEALTHY_ENGINES = {
    'analytics': 8100,
    'risk': 8200,
    'factor': 8300,
    'ml': 8400,
    'websocket': 8600,
    'marketdata': 8800,
    'collateral': 9000
}

class MessageBusLoadTester:
    def __init__(self):
        self.session = None
        self.results = {}
        
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=10)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def benchmark_engine_response_time(self, engine_name: str, port: int, iterations: int = 100) -> Dict:
        """Benchmark response times for a specific engine."""
        response_times = []
        errors = 0
        
        for _ in range(iterations):
            try:
                start_time = time.perf_counter()
                async with self.session.get(f"http://localhost:{port}/health") as response:
                    end_time = time.perf_counter()
                    if response.status == 200:
                        response_times.append((end_time - start_time) * 1000)  # Convert to ms
                    else:
                        errors += 1
            except:
                errors += 1
        
        if response_times:
            return {
                'engine': engine_name,
                'port': port,
                'iterations': iterations,
                'avg_response_time_ms': round(statistics.mean(response_times), 3),
                'min_response_time_ms': round(min(response_times), 3),
                'max_response_time_ms': round(max(response_times), 3),
                'median_response_time_ms': round(statistics.median(response_times), 3),
                'std_dev_ms': round(statistics.stdev(response_times), 3) if len(response_times) > 1 else 0,
                'success_rate': round((len(response_times) / iterations) * 100, 2),
                'errors': errors
            }
        else:
            return {
                'engine': engine_name,
                'port': port,
                'error': 'No successful responses',
                'errors': errors
            }
    
    async def test_concurrent_load(self, concurrent_requests: int = 50) -> Dict:
        """Test system under concurrent load."""
        logger.info(f"Testing concurrent load with {concurrent_requests} simultaneous requests...")
        
        tasks = []
        engines = list(HEALTHY_ENGINES.items())
        
        # Create concurrent requests across all engines
        for i in range(concurrent_requests):
            engine_name, port = engines[i % len(engines)]
            task = self.session.get(f"http://localhost:{port}/health")
            tasks.append((engine_name, port, task))
        
        start_time = time.perf_counter()
        results = []
        
        try:
            # Execute all requests concurrently
            async_tasks = [task for _, _, task in tasks]
            responses = await asyncio.gather(*async_tasks, return_exceptions=True)
            end_time = time.perf_counter()
            
            # Process results
            successful = 0
            failed = 0
            response_times = []
            
            for i, (engine_name, port, _) in enumerate(tasks):
                response = responses[i]
                if isinstance(response, Exception):
                    failed += 1
                else:
                    successful += 1
                    # Note: Individual response times not easily measured in gather()
                    # Using overall time as approximation
            
            total_time = (end_time - start_time) * 1000
            
            return {
                'concurrent_requests': concurrent_requests,
                'total_time_ms': round(total_time, 2),
                'avg_time_per_request_ms': round(total_time / concurrent_requests, 2),
                'requests_per_second': round(concurrent_requests / (total_time / 1000), 2),
                'successful_requests': successful,
                'failed_requests': failed,
                'success_rate': round((successful / concurrent_requests) * 100, 2)
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'concurrent_requests': concurrent_requests
            }
    
    async def test_messagebus_routing_performance(self) -> Dict:
        """Test messagebus routing performance by checking dual bus connectivity."""
        logger.info("Testing messagebus routing performance...")
        
        results = {}
        for engine_name, port in HEALTHY_ENGINES.items():
            try:
                start_time = time.perf_counter()
                async with self.session.get(f"http://localhost:{port}/health") as response:
                    end_time = time.perf_counter()
                    
                    if response.status == 200:
                        data = await response.json()
                        response_time = (end_time - start_time) * 1000
                        
                        # Check for dual messagebus indicators
                        dual_bus_connected = data.get('dual_messagebus_connected', False)
                        marketdata_bus = data.get('marketdata_bus', 'unknown')
                        engine_logic_bus = data.get('engine_logic_bus', 'unknown')
                        
                        results[engine_name] = {
                            'response_time_ms': round(response_time, 3),
                            'dual_messagebus_connected': dual_bus_connected,
                            'marketdata_bus_port': marketdata_bus,
                            'engine_logic_bus_port': engine_logic_bus,
                            'architecture': data.get('architecture', 'unknown'),
                            'status': 'healthy'
                        }
                    else:
                        results[engine_name] = {
                            'status': 'unhealthy',
                            'http_status': response.status
                        }
                        
            except Exception as e:
                results[engine_name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return results
    
    async def run_comprehensive_performance_test(self) -> Dict:
        """Run comprehensive performance testing suite."""
        logger.info("Starting comprehensive performance testing...")
        
        test_results = {
            'test_timestamp': datetime.now().isoformat(),
            'test_duration_seconds': 0,
            'healthy_engines': list(HEALTHY_ENGINES.keys()),
            'engine_count': len(HEALTHY_ENGINES)
        }
        
        overall_start = time.perf_counter()
        
        # 1. Individual engine response time benchmarks
        logger.info("1. Benchmarking individual engine response times...")
        benchmark_results = {}
        for engine_name, port in HEALTHY_ENGINES.items():
            benchmark_results[engine_name] = await self.benchmark_engine_response_time(
                engine_name, port, iterations=50
            )
        test_results['individual_benchmarks'] = benchmark_results
        
        # 2. Concurrent load testing
        logger.info("2. Testing concurrent load...")
        concurrent_tests = []
        for load in [10, 25, 50, 100]:
            result = await self.test_concurrent_load(load)
            result['load_level'] = load
            concurrent_tests.append(result)
        test_results['concurrent_load_tests'] = concurrent_tests
        
        # 3. MessageBus routing performance
        logger.info("3. Testing messagebus routing performance...")
        routing_results = await self.test_messagebus_routing_performance()
        test_results['messagebus_routing'] = routing_results
        
        # 4. Calculate overall performance metrics
        overall_end = time.perf_counter()
        test_results['test_duration_seconds'] = round(overall_end - overall_start, 2)
        
        # Calculate aggregate metrics
        all_response_times = []
        for engine_data in benchmark_results.values():
            if 'avg_response_time_ms' in engine_data:
                all_response_times.append(engine_data['avg_response_time_ms'])
        
        if all_response_times:
            test_results['aggregate_metrics'] = {
                'avg_response_time_ms': round(statistics.mean(all_response_times), 3),
                'min_response_time_ms': round(min(all_response_times), 3),
                'max_response_time_ms': round(max(all_response_times), 3),
                'median_response_time_ms': round(statistics.median(all_response_times), 3),
                'total_engines_tested': len(all_response_times)
            }
        
        # MessageBus health summary
        dual_bus_engines = sum(1 for engine_data in routing_results.values() 
                              if engine_data.get('dual_messagebus_connected', False))
        test_results['messagebus_summary'] = {
            'dual_bus_connected_engines': dual_bus_engines,
            'total_tested_engines': len(routing_results),
            'dual_bus_adoption_rate': round((dual_bus_engines / len(routing_results)) * 100, 2)
        }
        
        logger.info(f"Performance testing completed in {test_results['test_duration_seconds']}s")
        return test_results

async def main():
    """Run the comprehensive performance test."""
    async with MessageBusLoadTester() as tester:
        results = await tester.run_comprehensive_performance_test()
        
        # Save results
        timestamp = int(time.time())
        filename = f"messagebus_performance_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print("\n" + "="*80)
        print("MESSAGEBUS PERFORMANCE AND LOAD TEST RESULTS")
        print("="*80)
        
        print(f"Test Duration: {results['test_duration_seconds']}s")
        print(f"Engines Tested: {results['engine_count']}")
        
        # Individual benchmarks
        if 'aggregate_metrics' in results:
            metrics = results['aggregate_metrics']
            print(f"\n📊 AGGREGATE RESPONSE TIME METRICS:")
            print(f"  Average: {metrics['avg_response_time_ms']}ms")
            print(f"  Median:  {metrics['median_response_time_ms']}ms")
            print(f"  Min:     {metrics['min_response_time_ms']}ms")
            print(f"  Max:     {metrics['max_response_time_ms']}ms")
        
        # Concurrent load results
        if 'concurrent_load_tests' in results:
            print(f"\n⚡ CONCURRENT LOAD TEST RESULTS:")
            for test in results['concurrent_load_tests']:
                load = test['load_level']
                rps = test.get('requests_per_second', 'N/A')
                success = test.get('success_rate', 'N/A')
                print(f"  {load} concurrent requests: {rps} RPS, {success}% success rate")
        
        # MessageBus status
        if 'messagebus_summary' in results:
            mb_summary = results['messagebus_summary']
            print(f"\n🚌 DUAL MESSAGEBUS STATUS:")
            print(f"  Connected Engines: {mb_summary['dual_bus_connected_engines']}/{mb_summary['total_tested_engines']}")
            print(f"  Adoption Rate: {mb_summary['dual_bus_adoption_rate']}%")
        
        # Top performers
        if 'individual_benchmarks' in results:
            benchmarks = [(name, data) for name, data in results['individual_benchmarks'].items() 
                         if 'avg_response_time_ms' in data]
            benchmarks.sort(key=lambda x: x[1]['avg_response_time_ms'])
            
            print(f"\n🏆 TOP PERFORMING ENGINES:")
            for i, (engine, data) in enumerate(benchmarks[:3], 1):
                avg_time = data['avg_response_time_ms']
                success_rate = data['success_rate']
                print(f"  {i}. {engine}: {avg_time}ms avg, {success_rate}% success")
        
        print(f"\n💾 Full results saved to: {filename}")
        print("="*80)
        
        return results

if __name__ == "__main__":
    asyncio.run(main())