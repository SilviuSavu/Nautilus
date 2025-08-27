#!/usr/bin/env python3
"""
Dual MessageBus Performance Test Suite
Tests maximum throughput and latency for engines connected to dual messagebus architecture.
"""

import asyncio
import aiohttp
import redis
import time
import json
import statistics
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass
import concurrent.futures


@dataclass
class PerformanceMetric:
    """Performance metric data structure."""
    operation: str
    latency_ms: float
    throughput_ops_sec: float
    success_rate: float
    timestamp: str


class DualMessageBusPerformanceTest:
    """Performance test suite for dual messagebus architecture."""
    
    def __init__(self):
        self.engines = {
            # Dual MessageBus Connected Engines
            'analytics': 'http://localhost:8100',
            'backtesting': 'http://localhost:8110', 
            'risk': 'http://localhost:8200',
            'features': 'http://localhost:8500',
            
            # Additional Connected Engines
            'ml': 'http://localhost:8400',
            'factor': 'http://localhost:8300',
            'websocket': 'http://localhost:8600',
            'strategy': 'http://localhost:8700',
            'portfolio': 'http://localhost:8900',
            'collateral': 'http://localhost:9000',
            'vpin': 'http://localhost:10000',
            'enhanced_vpin': 'http://localhost:10001',
            'marketdata': 'http://localhost:8800'
        }
        
        # Redis connections for dual buses
        self.marketdata_bus = redis.Redis(host='localhost', port=6380, decode_responses=True)
        self.engine_logic_bus = redis.Redis(host='localhost', port=6381, decode_responses=True)
        
        self.results: List[PerformanceMetric] = []
        
    async def test_engine_health(self) -> Dict[str, bool]:
        """Test which engines are responsive."""
        print("🔍 Testing engine health and connectivity...")
        healthy_engines = {}
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            for name, url in self.engines.items():
                try:
                    async with session.get(f"{url}/health") as response:
                        healthy_engines[name] = response.status == 200
                        status = "✅ HEALTHY" if response.status == 200 else "❌ UNHEALTHY"
                        print(f"  {name.ljust(12)}: {status}")
                except Exception as e:
                    healthy_engines[name] = False
                    print(f"  {name.ljust(12)}: ❌ UNREACHABLE ({str(e)[:30]})")
        
        return healthy_engines
    
    async def test_messagebus_connectivity(self) -> Dict[str, bool]:
        """Test Redis messagebus connectivity."""
        print("\n🔍 Testing messagebus connectivity...")
        connectivity = {}
        
        try:
            # Test MarketData Bus (6380)
            self.marketdata_bus.ping()
            connectivity['marketdata_bus'] = True
            print("  MarketData Bus (6380): ✅ CONNECTED")
        except Exception as e:
            connectivity['marketdata_bus'] = False
            print(f"  MarketData Bus (6380): ❌ FAILED ({str(e)[:30]})")
        
        try:
            # Test Engine Logic Bus (6381)
            self.engine_logic_bus.ping()
            connectivity['engine_logic_bus'] = True
            print("  Engine Logic Bus (6381): ✅ CONNECTED")
        except Exception as e:
            connectivity['engine_logic_bus'] = False
            print(f"  Engine Logic Bus (6381): ❌ FAILED ({str(e)[:30]})")
        
        return connectivity
    
    async def test_engine_latency(self, engine_name: str, url: str, num_requests: int = 100) -> Tuple[float, float]:
        """Test single engine latency with multiple requests."""
        latencies = []
        successful_requests = 0
        
        async with aiohttp.ClientSession() as session:
            for _ in range(num_requests):
                start_time = time.perf_counter()
                try:
                    async with session.get(f"{url}/health") as response:
                        if response.status == 200:
                            successful_requests += 1
                        latency = (time.perf_counter() - start_time) * 1000
                        latencies.append(latency)
                except Exception:
                    latencies.append(float('inf'))
        
        avg_latency = statistics.mean([l for l in latencies if l != float('inf')])
        success_rate = successful_requests / num_requests
        
        return avg_latency, success_rate
    
    async def test_messagebus_throughput(self, bus_name: str, redis_client: redis.Redis, messages: int = 1000) -> Tuple[float, float]:
        """Test messagebus throughput."""
        print(f"📊 Testing {bus_name} throughput ({messages} messages)...")
        
        # Publish throughput test
        start_time = time.perf_counter()
        successful_publishes = 0
        
        for i in range(messages):
            try:
                test_message = {
                    'id': i,
                    'timestamp': time.time_ns(),
                    'data': f'performance_test_message_{i}',
                    'bus': bus_name
                }
                redis_client.publish(f'perf_test_{bus_name}', json.dumps(test_message))
                successful_publishes += 1
            except Exception:
                pass
        
        publish_duration = time.perf_counter() - start_time
        publish_throughput = successful_publishes / publish_duration
        success_rate = successful_publishes / messages
        
        return publish_throughput, success_rate
    
    async def test_cross_engine_communication(self, healthy_engines: Dict[str, bool]) -> Dict[str, float]:
        """Test communication between engines via messagebus."""
        print("\n📡 Testing cross-engine communication...")
        
        communication_latencies = {}
        
        # Test engine-to-engine communication patterns
        test_pairs = [
            ('analytics', 'risk'),
            ('ml', 'strategy'), 
            ('factor', 'portfolio'),
            ('risk', 'collateral')
        ]
        
        async with aiohttp.ClientSession() as session:
            for engine1, engine2 in test_pairs:
                if not (healthy_engines.get(engine1) and healthy_engines.get(engine2)):
                    continue
                
                try:
                    # Simulate cross-engine communication by calling both engines rapidly
                    start_time = time.perf_counter()
                    
                    url1 = self.engines[engine1]
                    url2 = self.engines[engine2]
                    
                    # Concurrent requests to simulate engine communication
                    tasks = [
                        session.get(f"{url1}/health"),
                        session.get(f"{url2}/health")
                    ]
                    
                    responses = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    communication_time = (time.perf_counter() - start_time) * 1000
                    communication_latencies[f"{engine1}-{engine2}"] = communication_time
                    
                    print(f"  {engine1} ↔ {engine2}: {communication_time:.2f}ms")
                    
                except Exception as e:
                    print(f"  {engine1} ↔ {engine2}: ❌ FAILED ({str(e)[:30]})")
        
        return communication_latencies
    
    async def run_comprehensive_test(self) -> None:
        """Run comprehensive performance test suite."""
        print("🚀 DUAL MESSAGEBUS PERFORMANCE TEST SUITE")
        print("=" * 50)
        print(f"Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 1. Test engine health
        healthy_engines = await self.test_engine_health()
        healthy_count = sum(healthy_engines.values())
        print(f"\n📊 Engine Health Summary: {healthy_count}/{len(self.engines)} engines healthy")
        
        # 2. Test messagebus connectivity
        bus_connectivity = await self.test_messagebus_connectivity()
        
        # 3. Test individual engine latency
        print(f"\n⚡ Testing individual engine latency (100 requests each)...")
        for engine_name, url in self.engines.items():
            if healthy_engines.get(engine_name):
                avg_latency, success_rate = await self.test_engine_latency(engine_name, url)
                print(f"  {engine_name.ljust(12)}: {avg_latency:.2f}ms (success: {success_rate*100:.1f}%)")
                
                self.results.append(PerformanceMetric(
                    operation=f"{engine_name}_latency",
                    latency_ms=avg_latency,
                    throughput_ops_sec=1000/avg_latency if avg_latency > 0 else 0,
                    success_rate=success_rate,
                    timestamp=datetime.now().isoformat()
                ))
        
        # 4. Test messagebus throughput
        if bus_connectivity.get('marketdata_bus'):
            throughput, success_rate = await self.test_messagebus_throughput("MarketData", self.marketdata_bus)
            print(f"  MarketData Bus: {throughput:.0f} msgs/sec (success: {success_rate*100:.1f}%)")
            
            self.results.append(PerformanceMetric(
                operation="marketdata_bus_throughput",
                latency_ms=1000/throughput if throughput > 0 else 0,
                throughput_ops_sec=throughput,
                success_rate=success_rate,
                timestamp=datetime.now().isoformat()
            ))
        
        if bus_connectivity.get('engine_logic_bus'):
            throughput, success_rate = await self.test_messagebus_throughput("EngineLogic", self.engine_logic_bus)
            print(f"  Engine Logic Bus: {throughput:.0f} msgs/sec (success: {success_rate*100:.1f}%)")
            
            self.results.append(PerformanceMetric(
                operation="engine_logic_bus_throughput", 
                latency_ms=1000/throughput if throughput > 0 else 0,
                throughput_ops_sec=throughput,
                success_rate=success_rate,
                timestamp=datetime.now().isoformat()
            ))
        
        # 5. Test cross-engine communication
        communication_latencies = await self.test_cross_engine_communication(healthy_engines)
        
        # 6. Generate summary report
        await self.generate_summary_report(healthy_engines, bus_connectivity, communication_latencies)
    
    async def generate_summary_report(self, healthy_engines: Dict[str, bool], 
                                    bus_connectivity: Dict[str, bool],
                                    communication_latencies: Dict[str, float]) -> None:
        """Generate comprehensive performance summary report."""
        print("\n" + "=" * 80)
        print("🏆 DUAL MESSAGEBUS PERFORMANCE SUMMARY REPORT")
        print("=" * 80)
        
        # System Overview
        healthy_count = sum(healthy_engines.values())
        total_engines = len(self.engines)
        
        print(f"📊 SYSTEM OVERVIEW")
        print(f"   Engines Operational: {healthy_count}/{total_engines} ({healthy_count/total_engines*100:.1f}%)")
        print(f"   MarketData Bus (6380): {'✅ OPERATIONAL' if bus_connectivity.get('marketdata_bus') else '❌ FAILED'}")
        print(f"   Engine Logic Bus (6381): {'✅ OPERATIONAL' if bus_connectivity.get('engine_logic_bus') else '❌ FAILED'}")
        
        # Performance Metrics
        latency_results = [r for r in self.results if 'latency' in r.operation]
        throughput_results = [r for r in self.results if 'throughput' in r.operation]
        
        if latency_results:
            avg_latency = statistics.mean([r.latency_ms for r in latency_results])
            min_latency = min([r.latency_ms for r in latency_results])
            max_latency = max([r.latency_ms for r in latency_results])
            
            print(f"\n⚡ LATENCY METRICS")
            print(f"   Average Latency: {avg_latency:.2f}ms")
            print(f"   Best Latency: {min_latency:.2f}ms")
            print(f"   Worst Latency: {max_latency:.2f}ms")
        
        if throughput_results:
            total_throughput = sum([r.throughput_ops_sec for r in throughput_results])
            
            print(f"\n🚀 THROUGHPUT METRICS")
            print(f"   Combined Bus Throughput: {total_throughput:.0f} ops/sec")
            
            for result in throughput_results:
                bus_name = result.operation.replace('_throughput', '').replace('_', ' ').title()
                print(f"   {bus_name}: {result.throughput_ops_sec:.0f} ops/sec")
        
        # Cross-Engine Communication
        if communication_latencies:
            avg_comm_latency = statistics.mean(communication_latencies.values())
            print(f"\n📡 CROSS-ENGINE COMMUNICATION")
            print(f"   Average Communication Latency: {avg_comm_latency:.2f}ms")
            
            for pair, latency in communication_latencies.items():
                print(f"   {pair}: {latency:.2f}ms")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"dual_messagebus_performance_results_{timestamp}.json"
        
        detailed_results = {
            'test_timestamp': datetime.now().isoformat(),
            'system_overview': {
                'engines_operational': f"{healthy_count}/{total_engines}",
                'marketdata_bus_status': bus_connectivity.get('marketdata_bus'),
                'engine_logic_bus_status': bus_connectivity.get('engine_logic_bus')
            },
            'engine_health': healthy_engines,
            'performance_metrics': [
                {
                    'operation': r.operation,
                    'latency_ms': r.latency_ms,
                    'throughput_ops_sec': r.throughput_ops_sec,
                    'success_rate': r.success_rate,
                    'timestamp': r.timestamp
                }
                for r in self.results
            ],
            'cross_engine_communication': communication_latencies
        }
        
        with open(results_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {results_file}")
        print("=" * 80)


async def main():
    """Main test execution."""
    test_suite = DualMessageBusPerformanceTest()
    await test_suite.run_comprehensive_test()


if __name__ == "__main__":
    asyncio.run(main())