#!/usr/bin/env python3
"""
Dual MessageBus Performance Testing Suite
Tests MarketData Bus (6380) vs Engine Logic Bus (6381) performance
"""

import asyncio
import time
import json
import redis
import statistics
from typing import Dict, List, Tuple
from datetime import datetime, timezone

class MessageBusPerformanceTester:
    def __init__(self):
        # MarketData Bus (6380) - Neural Engine optimized
        self.marketdata_redis = redis.Redis(host='localhost', port=6380, decode_responses=True)
        
        # Engine Logic Bus (6381) - Metal GPU optimized  
        self.engine_logic_redis = redis.Redis(host='localhost', port=6381, decode_responses=True)
        
        self.test_results = {}
        
    def generate_test_data(self, size: str = "small") -> Dict:
        """Generate test data of different sizes"""
        base_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": "AAPL",
            "price": 185.67,
            "volume": 1000,
            "bid": 185.65,
            "ask": 185.68
        }
        
        if size == "small":
            return base_data
        elif size == "medium":
            # Add more fields for medium payload
            base_data.update({
                "level2_bids": [[185.65, 500], [185.64, 1000], [185.63, 750]],
                "level2_asks": [[185.68, 300], [185.69, 800], [185.70, 600]],
                "market_status": "open",
                "session": "regular"
            })
            return base_data
        elif size == "large":
            # Large payload with extensive market data
            base_data.update({
                "level2_bids": [[185.65 - i*0.01, 100 + i*50] for i in range(20)],
                "level2_asks": [[185.68 + i*0.01, 100 + i*50] for i in range(20)],
                "market_status": "open",
                "session": "regular",
                "indicators": {
                    "rsi": 67.3,
                    "macd": 1.2,
                    "ema_20": 184.5,
                    "sma_50": 182.1,
                    "bollinger_upper": 188.2,
                    "bollinger_lower": 180.8
                },
                "historical_prices": [185.0 + i*0.1 for i in range(100)]
            })
            return base_data
    
    def test_redis_latency(self, redis_client: redis.Redis, bus_name: str, iterations: int = 1000) -> Dict:
        """Test basic Redis SET/GET latency"""
        print(f"\n🔧 Testing {bus_name} basic latency ({iterations} iterations)...")
        
        set_times = []
        get_times = []
        
        test_key = f"perf_test_{bus_name.lower().replace(' ', '_')}"
        test_value = json.dumps(self.generate_test_data("small"))
        
        # Test SET operations
        for i in range(iterations):
            start_time = time.perf_counter()
            redis_client.set(f"{test_key}_{i}", test_value)
            end_time = time.perf_counter()
            set_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        # Test GET operations
        for i in range(iterations):
            start_time = time.perf_counter()
            redis_client.get(f"{test_key}_{i}")
            end_time = time.perf_counter()
            get_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        # Cleanup
        for i in range(iterations):
            redis_client.delete(f"{test_key}_{i}")
        
        return {
            "set_avg_ms": statistics.mean(set_times),
            "set_median_ms": statistics.median(set_times),
            "set_min_ms": min(set_times),
            "set_max_ms": max(set_times),
            "get_avg_ms": statistics.mean(get_times),
            "get_median_ms": statistics.median(get_times),
            "get_min_ms": min(get_times),
            "get_max_ms": max(get_times)
        }
    
    def test_pub_sub_latency(self, redis_client: redis.Redis, bus_name: str, iterations: int = 100) -> Dict:
        """Test Pub/Sub latency"""
        print(f"\n📡 Testing {bus_name} Pub/Sub latency ({iterations} iterations)...")
        
        channel = f"test_channel_{bus_name.lower().replace(' ', '_')}"
        latencies = []
        
        # Create subscriber
        pubsub = redis_client.pubsub()
        pubsub.subscribe(channel)
        
        # Skip the subscription confirmation message
        pubsub.get_message(timeout=1)
        
        for i in range(iterations):
            # Prepare message with timestamp
            message_data = self.generate_test_data("small")
            message_data["test_id"] = i
            message_data["send_time"] = time.perf_counter()
            message = json.dumps(message_data)
            
            # Publish message
            redis_client.publish(channel, message)
            
            # Receive message and calculate latency
            received_msg = pubsub.get_message(timeout=1)
            if received_msg and received_msg['data']:
                receive_time = time.perf_counter()
                try:
                    received_data = json.loads(received_msg['data'])
                    send_time = received_data.get('send_time', 0)
                    latency_ms = (receive_time - send_time) * 1000
                    latencies.append(latency_ms)
                except json.JSONDecodeError:
                    continue
        
        pubsub.unsubscribe(channel)
        pubsub.close()
        
        if latencies:
            return {
                "avg_latency_ms": statistics.mean(latencies),
                "median_latency_ms": statistics.median(latencies),
                "min_latency_ms": min(latencies),
                "max_latency_ms": max(latencies),
                "successful_messages": len(latencies),
                "total_messages": iterations
            }
        else:
            return {"error": "No successful pub/sub messages"}
    
    def test_throughput(self, redis_client: redis.Redis, bus_name: str, duration_seconds: int = 10) -> Dict:
        """Test maximum throughput"""
        print(f"\n⚡ Testing {bus_name} throughput ({duration_seconds} seconds)...")
        
        test_data_small = json.dumps(self.generate_test_data("small"))
        test_data_medium = json.dumps(self.generate_test_data("medium"))
        test_data_large = json.dumps(self.generate_test_data("large"))
        
        results = {}
        
        for size, data in [("small", test_data_small), ("medium", test_data_medium), ("large", test_data_large)]:
            operations = 0
            start_time = time.time()
            end_time = start_time + duration_seconds
            
            while time.time() < end_time:
                redis_client.set(f"throughput_test_{size}_{operations}", data)
                operations += 1
            
            actual_duration = time.time() - start_time
            ops_per_second = operations / actual_duration
            
            results[f"{size}_payload"] = {
                "operations": operations,
                "duration_seconds": actual_duration,
                "ops_per_second": ops_per_second,
                "payload_size_bytes": len(data.encode('utf-8'))
            }
            
            # Cleanup
            redis_client.delete(f"throughput_test_{size}_{operations}")
        
        return results
    
    def test_stream_performance(self, redis_client: redis.Redis, bus_name: str, iterations: int = 100) -> Dict:
        """Test Redis Streams performance"""
        print(f"\n🌊 Testing {bus_name} Streams performance ({iterations} iterations)...")
        
        stream_name = f"test_stream_{bus_name.lower().replace(' ', '_')}"
        latencies = []
        
        for i in range(iterations):
            test_data = self.generate_test_data("small")
            test_data["test_id"] = i
            
            start_time = time.perf_counter()
            
            # Add to stream
            stream_id = redis_client.xadd(stream_name, test_data)
            
            # Read from stream
            messages = redis_client.xrange(stream_name, min=stream_id, max=stream_id)
            
            end_time = time.perf_counter()
            
            if messages:
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
        
        # Cleanup
        redis_client.delete(stream_name)
        
        if latencies:
            return {
                "avg_latency_ms": statistics.mean(latencies),
                "median_latency_ms": statistics.median(latencies),
                "min_latency_ms": min(latencies),
                "max_latency_ms": max(latencies)
            }
        else:
            return {"error": "No successful stream operations"}
    
    def run_comprehensive_test(self):
        """Run comprehensive performance test on both message buses"""
        print("🚀 DUAL MESSAGEBUS PERFORMANCE TESTING SUITE")
        print("=" * 60)
        
        # Test MarketData Bus (6380)
        print(f"\n📊 TESTING MARKETDATA BUS (Port 6380) - Neural Engine Optimized")
        print("-" * 60)
        
        try:
            self.test_results["marketdata_bus"] = {
                "basic_latency": self.test_redis_latency(self.marketdata_redis, "MarketData Bus"),
                "pubsub_latency": self.test_pub_sub_latency(self.marketdata_redis, "MarketData Bus"),
                "throughput": self.test_throughput(self.marketdata_redis, "MarketData Bus"),
                "streams": self.test_stream_performance(self.marketdata_redis, "MarketData Bus")
            }
        except Exception as e:
            self.test_results["marketdata_bus"] = {"error": str(e)}
            print(f"❌ MarketData Bus test failed: {e}")
        
        # Test Engine Logic Bus (6381)
        print(f"\n⚙️ TESTING ENGINE LOGIC BUS (Port 6381) - Metal GPU Optimized")
        print("-" * 60)
        
        try:
            self.test_results["engine_logic_bus"] = {
                "basic_latency": self.test_redis_latency(self.engine_logic_redis, "Engine Logic Bus"),
                "pubsub_latency": self.test_pub_sub_latency(self.engine_logic_redis, "Engine Logic Bus"),
                "throughput": self.test_throughput(self.engine_logic_redis, "Engine Logic Bus"),
                "streams": self.test_stream_performance(self.engine_logic_redis, "Engine Logic Bus")
            }
        except Exception as e:
            self.test_results["engine_logic_bus"] = {"error": str(e)}
            print(f"❌ Engine Logic Bus test failed: {e}")
        
        # Display results
        self.display_results()
    
    def display_results(self):
        """Display comprehensive test results"""
        print("\n🏆 DUAL MESSAGEBUS PERFORMANCE RESULTS")
        print("=" * 80)
        
        for bus_name, results in self.test_results.items():
            if "error" in results:
                print(f"\n❌ {bus_name.upper()}: {results['error']}")
                continue
                
            print(f"\n📈 {bus_name.replace('_', ' ').upper()} PERFORMANCE:")
            print("-" * 50)
            
            # Basic Latency
            if "basic_latency" in results:
                basic = results["basic_latency"]
                print(f"🔧 Basic Operations:")
                print(f"   SET: {basic['set_avg_ms']:.3f}ms avg, {basic['set_min_ms']:.3f}ms min")
                print(f"   GET: {basic['get_avg_ms']:.3f}ms avg, {basic['get_min_ms']:.3f}ms min")
            
            # Pub/Sub Latency
            if "pubsub_latency" in results and "error" not in results["pubsub_latency"]:
                pubsub = results["pubsub_latency"]
                print(f"📡 Pub/Sub:")
                print(f"   Latency: {pubsub['avg_latency_ms']:.3f}ms avg, {pubsub['min_latency_ms']:.3f}ms min")
                print(f"   Success: {pubsub['successful_messages']}/{pubsub['total_messages']} messages")
            
            # Throughput
            if "throughput" in results:
                throughput = results["throughput"]
                for size, data in throughput.items():
                    print(f"⚡ Throughput ({size}):")
                    print(f"   {data['ops_per_second']:.0f} ops/sec ({data['payload_size_bytes']} bytes)")
            
            # Streams
            if "streams" in results and "error" not in results["streams"]:
                streams = results["streams"]
                print(f"🌊 Streams:")
                print(f"   Latency: {streams['avg_latency_ms']:.3f}ms avg, {streams['min_latency_ms']:.3f}ms min")
        
        # Performance comparison
        self.compare_performance()
    
    def compare_performance(self):
        """Compare performance between the two buses"""
        print(f"\n🔬 PERFORMANCE COMPARISON")
        print("=" * 50)
        
        try:
            md_basic = self.test_results["marketdata_bus"]["basic_latency"]
            el_basic = self.test_results["engine_logic_bus"]["basic_latency"]
            
            print(f"Basic SET Operations:")
            print(f"   MarketData Bus: {md_basic['set_avg_ms']:.3f}ms")
            print(f"   Engine Logic Bus: {el_basic['set_avg_ms']:.3f}ms")
            
            if md_basic['set_avg_ms'] < el_basic['set_avg_ms']:
                improvement = ((el_basic['set_avg_ms'] - md_basic['set_avg_ms']) / el_basic['set_avg_ms']) * 100
                print(f"   🏆 MarketData Bus is {improvement:.1f}% faster for SET operations")
            else:
                improvement = ((md_basic['set_avg_ms'] - el_basic['set_avg_ms']) / md_basic['set_avg_ms']) * 100
                print(f"   🏆 Engine Logic Bus is {improvement:.1f}% faster for SET operations")
        
        except KeyError:
            print("❌ Unable to compare - missing test data")

if __name__ == "__main__":
    tester = MessageBusPerformanceTester()
    tester.run_comprehensive_test()