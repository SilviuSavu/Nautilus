#!/usr/bin/env python3
"""
⚡ Agent Lightning: Fast Performance Validation
Quick validation of M4 Max ultra-performance optimizations
"""

import redis
import time
import statistics
import json
from datetime import datetime
import numpy as np

def quick_latency_test(host: str, port: int, name: str, target_us: float, num_tests: int = 500):
    """Quick latency benchmark"""
    print(f"🧪 {name} Quick Latency Test ({num_tests} tests)")
    
    try:
        r = redis.Redis(host=host, port=port, decode_responses=True)
        
        # Warm up
        for _ in range(50):
            r.ping()
        
        latencies = []
        for i in range(num_tests):
            start = time.time_ns()
            r.ping()
            end = time.time_ns()
            latency_us = (end - start) / 1000
            latencies.append(latency_us)
            
            if i % 100 == 0 and i > 0:
                time.sleep(0.001)  # Brief pause every 100
        
        avg = statistics.mean(latencies)
        min_lat = min(latencies)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        
        target_achieved = avg < target_us
        
        print(f"   Average: {avg:.1f}μs (Target: <{target_us}μs) {'✅' if target_achieved else '❌'}")
        print(f"   Min: {min_lat:.1f}μs | P95: {p95:.1f}μs | P99: {p99:.1f}μs")
        
        return {
            "average_us": avg,
            "min_us": min_lat,
            "p95_us": p95,
            "p99_us": p99,
            "target_achieved": target_achieved,
            "target_us": target_us
        }
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return {"error": str(e), "target_achieved": False}

def quick_throughput_test(host: str, port: int, name: str, duration_sec: int = 5):
    """Quick throughput benchmark"""
    print(f"🚀 {name} Quick Throughput Test ({duration_sec}s)")
    
    try:
        r = redis.Redis(host=host, port=port, decode_responses=True)
        
        start_time = time.time()
        end_time = start_time + duration_sec
        operations = 0
        
        while time.time() < end_time:
            r.ping()
            operations += 1
        
        actual_duration = time.time() - start_time
        throughput = operations / actual_duration
        
        print(f"   Operations: {operations:,} in {actual_duration:.2f}s")
        print(f"   Throughput: {throughput:,.0f} ops/sec")
        
        return {
            "operations": operations,
            "duration_sec": actual_duration,
            "throughput_ops_sec": throughput
        }
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return {"error": str(e)}

def main():
    """Execute lightning-fast performance validation"""
    print("⚡ AGENT LIGHTNING: LIGHTNING-FAST PERFORMANCE VALIDATION")
    print("🎯 Quick validation of M4 Max ultra-performance optimizations")
    print("=" * 70)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "tests": {}
    }
    
    # Test 1: MarketData Bus (Neural Engine optimized)
    print("\n🔥 MarketData Bus Performance (Neural Engine + Unified Memory)")
    print("   Target: Sub-100μs latency | Hardware: Neural Engine 38 TOPS")
    
    md_latency = quick_latency_test("localhost", 6380, "MarketData Bus", 100.0, 500)
    md_throughput = quick_throughput_test("localhost", 6380, "MarketData Bus", 5)
    
    results["tests"]["marketdata_bus"] = {
        "latency": md_latency,
        "throughput": md_throughput
    }
    
    print("\n" + "-" * 50)
    
    # Test 2: Engine Logic Bus (Metal GPU optimized)
    print("\n🔥 Engine Logic Bus Performance (Metal GPU + Performance Cores)")
    print("   Target: Sub-50μs latency | Hardware: Metal GPU 40 cores")
    
    el_latency = quick_latency_test("localhost", 6381, "Engine Logic Bus", 50.0, 500)
    el_throughput = quick_throughput_test("localhost", 6381, "Engine Logic Bus", 5)
    
    results["tests"]["engine_logic_bus"] = {
        "latency": el_latency,
        "throughput": el_throughput
    }
    
    # Performance Summary
    print("\n" + "=" * 70)
    print("🏆 LIGHTNING-FAST PERFORMANCE SUMMARY")
    print("=" * 70)
    
    md_target_met = md_latency.get("target_achieved", False)
    el_target_met = el_latency.get("target_achieved", False)
    
    combined_throughput = (md_throughput.get("throughput_ops_sec", 0) + 
                          el_throughput.get("throughput_ops_sec", 0))
    
    print(f"📊 LATENCY RESULTS:")
    print(f"   MarketData Bus: {md_latency.get('average_us', 0):.1f}μs (Target: <100μs) {'✅' if md_target_met else '❌'}")
    print(f"   Engine Logic Bus: {el_latency.get('average_us', 0):.1f}μs (Target: <50μs) {'✅' if el_target_met else '❌'}")
    
    print(f"\n🚀 THROUGHPUT RESULTS:")
    print(f"   MarketData Bus: {md_throughput.get('throughput_ops_sec', 0):,.0f} ops/sec")
    print(f"   Engine Logic Bus: {el_throughput.get('throughput_ops_sec', 0):,.0f} ops/sec")
    print(f"   Combined: {combined_throughput:,.0f} ops/sec (Target: 1M+ ops/sec)")
    
    million_msg_target = combined_throughput >= 1000000
    
    print(f"\n⚡ PERFORMANCE GRADE:")
    if md_target_met and el_target_met and million_msg_target:
        grade = "A+ THEORETICAL MAXIMUM"
        emoji = "🏆"
    elif (md_target_met or el_target_met) and combined_throughput >= 500000:
        grade = "A- EXCELLENT"
        emoji = "🥇"
    elif combined_throughput >= 100000:
        grade = "B+ GOOD"
        emoji = "🥈"
    else:
        grade = "C NEEDS OPTIMIZATION"
        emoji = "⚠️"
    
    print(f"   {emoji} OVERALL GRADE: {grade}")
    
    # Performance improvements vs baseline
    baseline_md = 258.0  # From DUAL_MESSAGEBUS_ARCHITECTURE.md
    baseline_el = 354.0  # From DUAL_MESSAGEBUS_ARCHITECTURE.md
    
    md_improvement = baseline_md / md_latency.get('average_us', baseline_md)
    el_improvement = baseline_el / el_latency.get('average_us', baseline_el)
    
    print(f"\n📈 PERFORMANCE IMPROVEMENTS:")
    print(f"   MarketData Bus: {md_improvement:.1f}x faster than baseline ({baseline_md}μs → {md_latency.get('average_us', 0):.1f}μs)")
    print(f"   Engine Logic Bus: {el_improvement:.1f}x faster than baseline ({baseline_el}μs → {el_latency.get('average_us', 0):.1f}μs)")
    
    # Save results
    results["performance_grade"] = grade
    results["targets_achieved"] = {
        "marketdata_sub_100us": md_target_met,
        "engine_logic_sub_50us": el_target_met,
        "combined_1m_ops": million_msg_target
    }
    results["improvements"] = {
        "marketdata_improvement_factor": md_improvement,
        "engine_logic_improvement_factor": el_improvement
    }
    
    results_file = f"lightning_fast_results_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    print("\n⚡ Agent Lightning: Lightning-fast validation complete!")
    
    return results

if __name__ == "__main__":
    main()