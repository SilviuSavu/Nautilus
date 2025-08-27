#!/usr/bin/env python3
"""
Redis Speed Test - Maximum Throughput Analysis
Tests all three Redis instances in the Nautilus dual messagebus architecture
"""

import asyncio
import time
import json
import random
import string
import statistics
from typing import Dict, List, Tuple
import redis.asyncio as redis
from concurrent.futures import ThreadPoolExecutor
import sys

class RedisSpeedTester:
    """Comprehensive Redis performance testing"""
    
    def __init__(self):
        self.results = {}
        
        # Redis configurations
        self.redis_configs = {
            "main_redis": {"host": "localhost", "port": 6379, "db": 0},
            "marketdata_bus": {"host": "localhost", "port": 6380, "db": 0},
            "engine_logic_bus": {"host": "localhost", "port": 6381, "db": 0}
        }
        
    def generate_test_data(self, size_kb: int) -> str:
        """Generate test data of specified size in KB"""
        # Create realistic market data payload
        base_data = {
            "symbol": "".join(random.choices(string.ascii_uppercase, k=4)),
            "timestamp": time.time(),
            "price": round(random.uniform(100, 1000), 2),
            "volume": random.randint(1000000, 10000000),
            "bid": round(random.uniform(100, 1000), 2),
            "ask": round(random.uniform(100, 1000), 2),
            "market_depth": [
                {"level": i, "bid_price": round(random.uniform(100, 1000), 2), 
                 "bid_size": random.randint(100, 10000),
                 "ask_price": round(random.uniform(100, 1000), 2),
                 "ask_size": random.randint(100, 10000)} for i in range(10)
            ],
            "technical_indicators": {
                "sma_20": round(random.uniform(100, 1000), 2),
                "ema_12": round(random.uniform(100, 1000), 2),
                "rsi": round(random.uniform(0, 100), 2),
                "macd": round(random.uniform(-50, 50), 2)
            }
        }
        
        # Convert to JSON and pad to desired size
        json_data = json.dumps(base_data)
        current_size = len(json_data.encode('utf-8'))
        target_size = size_kb * 1024
        
        if current_size < target_size:
            padding_needed = target_size - current_size
            padding = "x" * padding_needed
            base_data["padding"] = padding
            json_data = json.dumps(base_data)
            
        return json_data
    
    async def test_redis_set_get(self, redis_client: redis.Redis, test_name: str, 
                                data_size_kb: int, operations: int) -> Dict:
        """Test Redis SET/GET operations"""
        print(f"🧪 Testing {test_name} - {data_size_kb}KB payloads, {operations} operations...")
        
        # Generate test data
        test_data = self.generate_test_data(data_size_kb)
        data_size_bytes = len(test_data.encode('utf-8'))
        
        # SET operations
        set_times = []
        start_time = time.perf_counter()
        
        for i in range(operations):
            op_start = time.perf_counter()
            await redis_client.set(f"test_key_{i}", test_data)
            op_end = time.perf_counter()
            set_times.append(op_end - op_start)
            
        set_total_time = time.perf_counter() - start_time
        
        # GET operations
        get_times = []
        start_time = time.perf_counter()
        
        for i in range(operations):
            op_start = time.perf_counter()
            await redis_client.get(f"test_key_{i}")
            op_end = time.perf_counter()
            get_times.append(op_end - op_start)
            
        get_total_time = time.perf_counter() - start_time
        
        # Calculate throughput
        total_data_bytes = operations * data_size_bytes
        total_data_gb = total_data_bytes / (1024 ** 3)
        
        set_throughput_gb_s = total_data_gb / set_total_time
        get_throughput_gb_s = total_data_gb / get_total_time
        
        # Cleanup
        for i in range(operations):
            await redis_client.delete(f"test_key_{i}")
        
        return {
            "test_name": test_name,
            "data_size_kb": data_size_kb,
            "operations": operations,
            "total_data_gb": round(total_data_gb, 4),
            "set_performance": {
                "total_time_s": round(set_total_time, 4),
                "avg_latency_ms": round(statistics.mean(set_times) * 1000, 4),
                "throughput_gb_s": round(set_throughput_gb_s, 4),
                "throughput_mb_s": round(set_throughput_gb_s * 1024, 2),
                "ops_per_second": round(operations / set_total_time, 2)
            },
            "get_performance": {
                "total_time_s": round(get_total_time, 4),
                "avg_latency_ms": round(statistics.mean(get_times) * 1000, 4),
                "throughput_gb_s": round(get_throughput_gb_s, 4),
                "throughput_mb_s": round(get_throughput_gb_s * 1024, 2),
                "ops_per_second": round(operations / get_total_time, 2)
            }
        }
    
    async def test_redis_streams(self, redis_client: redis.Redis, test_name: str,
                               data_size_kb: int, operations: int) -> Dict:
        """Test Redis Streams (XADD/XREAD) operations"""
        print(f"🌊 Testing {test_name} Streams - {data_size_kb}KB payloads, {operations} operations...")
        
        stream_name = f"speed_test_stream_{int(time.time())}"
        test_data = self.generate_test_data(data_size_kb)
        data_size_bytes = len(test_data.encode('utf-8'))
        
        # XADD operations
        add_times = []
        start_time = time.perf_counter()
        
        for i in range(operations):
            op_start = time.perf_counter()
            await redis_client.xadd(stream_name, {
                "message_type": "speed_test",
                "payload": test_data,
                "sequence": str(i),
                "timestamp": str(time.time())
            })
            op_end = time.perf_counter()
            add_times.append(op_end - op_start)
            
        add_total_time = time.perf_counter() - start_time
        
        # XREAD operations
        read_times = []
        start_time = time.perf_counter()
        
        # Read all messages
        op_start = time.perf_counter()
        messages = await redis_client.xrange(stream_name, min="-", max="+", count=operations)
        op_end = time.perf_counter()
        read_total_time = op_end - op_start
        
        # Calculate throughput
        total_data_bytes = operations * data_size_bytes
        total_data_gb = total_data_bytes / (1024 ** 3)
        
        add_throughput_gb_s = total_data_gb / add_total_time
        read_throughput_gb_s = total_data_gb / read_total_time
        
        # Cleanup
        await redis_client.delete(stream_name)
        
        return {
            "test_name": f"{test_name} (Streams)",
            "data_size_kb": data_size_kb,
            "operations": operations,
            "total_data_gb": round(total_data_gb, 4),
            "messages_processed": len(messages),
            "xadd_performance": {
                "total_time_s": round(add_total_time, 4),
                "avg_latency_ms": round(statistics.mean(add_times) * 1000, 4),
                "throughput_gb_s": round(add_throughput_gb_s, 4),
                "throughput_mb_s": round(add_throughput_gb_s * 1024, 2),
                "ops_per_second": round(operations / add_total_time, 2)
            },
            "xread_performance": {
                "total_time_s": round(read_total_time, 4),
                "throughput_gb_s": round(read_throughput_gb_s, 4),
                "throughput_mb_s": round(read_throughput_gb_s * 1024, 2),
                "messages_per_second": round(len(messages) / read_total_time, 2)
            }
        }
    
    async def test_concurrent_operations(self, redis_client: redis.Redis, test_name: str,
                                       data_size_kb: int, total_operations: int, 
                                       concurrency: int) -> Dict:
        """Test concurrent Redis operations"""
        print(f"⚡ Testing {test_name} Concurrent - {concurrency} workers, {total_operations} ops...")
        
        test_data = self.generate_test_data(data_size_kb)
        data_size_bytes = len(test_data.encode('utf-8'))
        ops_per_worker = total_operations // concurrency
        
        async def worker(worker_id: int):
            worker_times = []
            for i in range(ops_per_worker):
                key = f"concurrent_test_{worker_id}_{i}"
                op_start = time.perf_counter()
                await redis_client.set(key, test_data)
                await redis_client.get(key)
                await redis_client.delete(key)
                op_end = time.perf_counter()
                worker_times.append(op_end - op_start)
            return worker_times
        
        # Run concurrent workers
        start_time = time.perf_counter()
        tasks = [worker(i) for i in range(concurrency)]
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_time
        
        # Flatten all operation times
        all_times = [time for worker_times in results for time in worker_times]
        actual_operations = len(all_times)
        
        # Calculate throughput
        total_data_bytes = actual_operations * data_size_bytes
        total_data_gb = total_data_bytes / (1024 ** 3)
        throughput_gb_s = total_data_gb / total_time
        
        return {
            "test_name": f"{test_name} (Concurrent)",
            "data_size_kb": data_size_kb,
            "planned_operations": total_operations,
            "actual_operations": actual_operations,
            "concurrency": concurrency,
            "total_data_gb": round(total_data_gb, 4),
            "performance": {
                "total_time_s": round(total_time, 4),
                "avg_latency_ms": round(statistics.mean(all_times) * 1000, 4),
                "throughput_gb_s": round(throughput_gb_s, 4),
                "throughput_mb_s": round(throughput_gb_s * 1024, 2),
                "ops_per_second": round(actual_operations / total_time, 2)
            }
        }
    
    async def run_comprehensive_test(self):
        """Run comprehensive speed tests on all Redis instances"""
        print("🚀 Starting Comprehensive Redis Speed Test")
        print("=" * 60)
        
        all_results = []
        
        for redis_name, config in self.redis_configs.items():
            print(f"\n📡 Testing {redis_name.upper()} ({config['host']}:{config['port']})")
            print("-" * 40)
            
            try:
                # Connect to Redis
                redis_client = redis.Redis(**config)
                await redis_client.ping()
                print(f"✅ Connected to {redis_name}")
                
                # Test different payload sizes and operation counts
                test_scenarios = [
                    {"size_kb": 1, "operations": 10000},      # Small payloads, many ops
                    {"size_kb": 10, "operations": 5000},      # Medium payloads
                    {"size_kb": 100, "operations": 1000},     # Large payloads
                    {"size_kb": 1000, "operations": 500},     # Very large payloads
                ]
                
                for scenario in test_scenarios:
                    # SET/GET test
                    result = await self.test_redis_set_get(
                        redis_client, redis_name, 
                        scenario["size_kb"], scenario["operations"]
                    )
                    all_results.append(result)
                    
                    # Streams test
                    if scenario["size_kb"] <= 100:  # Limit streams test to smaller payloads
                        stream_result = await self.test_redis_streams(
                            redis_client, redis_name,
                            scenario["size_kb"], min(scenario["operations"], 1000)
                        )
                        all_results.append(stream_result)
                
                # Concurrent test
                concurrent_result = await self.test_concurrent_operations(
                    redis_client, redis_name, 10, 5000, 10
                )
                all_results.append(concurrent_result)
                
                await redis_client.aclose()
                
            except Exception as e:
                print(f"❌ Error testing {redis_name}: {e}")
                continue
        
        return all_results
    
    def analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze all test results and find maximum throughput"""
        print("\n🔬 ANALYSIS - Maximum Throughput Results")
        print("=" * 60)
        
        max_set_throughput = 0
        max_get_throughput = 0
        max_stream_write_throughput = 0
        max_stream_read_throughput = 0
        max_concurrent_throughput = 0
        
        best_set_test = None
        best_get_test = None
        best_stream_write_test = None
        best_stream_read_test = None
        best_concurrent_test = None
        
        for result in results:
            # SET operations
            if "set_performance" in result:
                throughput = result["set_performance"]["throughput_gb_s"]
                if throughput > max_set_throughput:
                    max_set_throughput = throughput
                    best_set_test = result
            
            # GET operations
            if "get_performance" in result:
                throughput = result["get_performance"]["throughput_gb_s"]
                if throughput > max_get_throughput:
                    max_get_throughput = throughput
                    best_get_test = result
            
            # Stream operations
            if "xadd_performance" in result:
                throughput = result["xadd_performance"]["throughput_gb_s"]
                if throughput > max_stream_write_throughput:
                    max_stream_write_throughput = throughput
                    best_stream_write_test = result
            
            if "xread_performance" in result:
                throughput = result["xread_performance"]["throughput_gb_s"]
                if throughput > max_stream_read_throughput:
                    max_stream_read_throughput = throughput
                    best_stream_read_test = result
            
            # Concurrent operations
            if "performance" in result and "Concurrent" in result["test_name"]:
                throughput = result["performance"]["throughput_gb_s"]
                if throughput > max_concurrent_throughput:
                    max_concurrent_throughput = throughput
                    best_concurrent_test = result
        
        # Print summary
        print(f"🏆 MAXIMUM THROUGHPUT ACHIEVED:")
        print(f"   SET Operations:     {max_set_throughput:.4f} GB/s ({max_set_throughput * 1024:.2f} MB/s)")
        print(f"   GET Operations:     {max_get_throughput:.4f} GB/s ({max_get_throughput * 1024:.2f} MB/s)")
        print(f"   Stream Write:       {max_stream_write_throughput:.4f} GB/s ({max_stream_write_throughput * 1024:.2f} MB/s)")
        print(f"   Stream Read:        {max_stream_read_throughput:.4f} GB/s ({max_stream_read_throughput * 1024:.2f} MB/s)")
        print(f"   Concurrent Ops:     {max_concurrent_throughput:.4f} GB/s ({max_concurrent_throughput * 1024:.2f} MB/s)")
        
        overall_max = max(max_set_throughput, max_get_throughput, 
                         max_stream_write_throughput, max_stream_read_throughput,
                         max_concurrent_throughput)
        
        print(f"\n🎯 OVERALL MAXIMUM: {overall_max:.4f} GB/s ({overall_max * 1024:.2f} MB/s)")
        
        return {
            "max_throughputs": {
                "set_gb_s": max_set_throughput,
                "get_gb_s": max_get_throughput,
                "stream_write_gb_s": max_stream_write_throughput,
                "stream_read_gb_s": max_stream_read_throughput,
                "concurrent_gb_s": max_concurrent_throughput,
                "overall_max_gb_s": overall_max
            },
            "best_performers": {
                "set": best_set_test,
                "get": best_get_test,
                "stream_write": best_stream_write_test,
                "stream_read": best_stream_read_test,
                "concurrent": best_concurrent_test
            }
        }
    
    def print_detailed_results(self, results: List[Dict]):
        """Print detailed results for all tests"""
        print("\n📊 DETAILED TEST RESULTS")
        print("=" * 60)
        
        for result in results:
            print(f"\n🧪 {result['test_name']} ({result['data_size_kb']}KB payloads)")
            print(f"   Operations: {result['operations']}")
            print(f"   Total Data: {result['total_data_gb']} GB")
            
            if "set_performance" in result:
                perf = result["set_performance"]
                print(f"   SET: {perf['throughput_gb_s']} GB/s ({perf['throughput_mb_s']} MB/s) - {perf['ops_per_second']} ops/s")
                
            if "get_performance" in result:
                perf = result["get_performance"]
                print(f"   GET: {perf['throughput_gb_s']} GB/s ({perf['throughput_mb_s']} MB/s) - {perf['ops_per_second']} ops/s")
                
            if "xadd_performance" in result:
                perf = result["xadd_performance"]
                print(f"   XADD: {perf['throughput_gb_s']} GB/s ({perf['throughput_mb_s']} MB/s) - {perf['ops_per_second']} ops/s")
                
            if "xread_performance" in result:
                perf = result["xread_performance"]
                print(f"   XREAD: {perf['throughput_gb_s']} GB/s ({perf['throughput_mb_s']} MB/s)")
                
            if "performance" in result and "Concurrent" in result['test_name']:
                perf = result["performance"]
                print(f"   CONCURRENT: {perf['throughput_gb_s']} GB/s ({perf['throughput_mb_s']} MB/s) - {perf['ops_per_second']} ops/s")

async def main():
    """Main test execution"""
    tester = RedisSpeedTester()
    
    try:
        # Run comprehensive tests
        results = await tester.run_comprehensive_test()
        
        # Print detailed results
        tester.print_detailed_results(results)
        
        # Analyze and show maximum throughput
        analysis = tester.analyze_results(results)
        
        # Save results to file
        output_file = f"redis_speed_test_results_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump({
                "test_results": results,
                "analysis": analysis,
                "test_timestamp": time.time(),
                "system_info": {
                    "python_version": sys.version,
                    "redis_configs": tester.redis_configs
                }
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")

if __name__ == "__main__":
    print("🚀 Redis Speed Test - Nautilus Dual MessageBus Architecture")
    print("Testing maximum throughput across all Redis instances...")
    asyncio.run(main())