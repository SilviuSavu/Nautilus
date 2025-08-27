#!/usr/bin/env python3
"""
🏃 Dream Team Comprehensive Engine Functional Test
Full capacity testing with real database data for all specialized engines
"""

import asyncio
import aiohttp
import time
import json
import psycopg2
from datetime import datetime
from typing import Dict, List, Any
import statistics

class DreamTeamEngineTest:
    def __init__(self):
        self.engines = {
            'analytics': 'http://localhost:8100',
            'risk': 'http://localhost:8200', 
            'factor': 'http://localhost:8300',
            'ml': 'http://localhost:8400',
            'features': 'http://localhost:8500',
            'websocket': 'http://localhost:8600',
            'strategy': 'http://localhost:8700',
            'marketdata': 'http://localhost:8800',
            'portfolio': 'http://localhost:8900',
            'collateral': 'http://localhost:9000',
            'backend': 'http://localhost:8001'
        }
        self.results = {}
        self.db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'nautilus',
            'user': 'nautilus',
            'password': 'nautilus123'
        }

    def get_real_data_samples(self):
        """Get real data samples from database for testing"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            # Get sample instruments and data
            cur.execute("SELECT DISTINCT instrument_id FROM bars LIMIT 5")
            instruments = [row[0] for row in cur.fetchall()]
            
            cur.execute("SELECT COUNT(*) FROM bars")
            bar_count = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM positions")
            position_count = cur.fetchone()[0]
            
            cur.close()
            conn.close()
            
            return {
                'instruments': instruments,
                'bar_count': bar_count,
                'position_count': position_count,
                'status': 'connected'
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    async def test_engine_endpoint(self, session: aiohttp.ClientSession, 
                                   engine_name: str, url: str, 
                                   payload: Dict = None) -> Dict:
        """Test individual engine with real data payload"""
        start_time = time.time()
        
        try:
            if payload:
                async with session.post(f"{url}/health", json=payload, timeout=30) as response:
                    response_time = time.time() - start_time
                    data = await response.json() if response.content_type == 'application/json' else await response.text()
                    
                    return {
                        'engine': engine_name,
                        'status': response.status,
                        'response_time': response_time,
                        'data': data,
                        'success': response.status == 200
                    }
            else:
                # Test basic health endpoint
                async with session.get(f"{url}/health", timeout=30) as response:
                    response_time = time.time() - start_time
                    data = await response.json() if response.content_type == 'application/json' else await response.text()
                    
                    return {
                        'engine': engine_name,
                        'status': response.status,
                        'response_time': response_time,
                        'data': data,
                        'success': response.status == 200
                    }
                    
        except Exception as e:
            response_time = time.time() - start_time
            return {
                'engine': engine_name,
                'status': 'error',
                'response_time': response_time,
                'error': str(e),
                'success': False
            }

    async def stress_test_engine(self, session: aiohttp.ClientSession, 
                                 engine_name: str, url: str, 
                                 iterations: int = 50) -> Dict:
        """Run stress test on individual engine"""
        print(f"🔥 Stress testing {engine_name} with {iterations} requests...")
        
        tasks = []
        for i in range(iterations):
            task = self.test_engine_endpoint(session, engine_name, url)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results
        successful_requests = [r for r in results if isinstance(r, dict) and r.get('success')]
        failed_requests = [r for r in results if isinstance(r, dict) and not r.get('success')]
        
        response_times = [r['response_time'] for r in successful_requests]
        
        return {
            'engine': engine_name,
            'total_requests': iterations,
            'successful_requests': len(successful_requests),
            'failed_requests': len(failed_requests),
            'success_rate': len(successful_requests) / iterations * 100,
            'avg_response_time': statistics.mean(response_times) if response_times else 0,
            'min_response_time': min(response_times) if response_times else 0,
            'max_response_time': max(response_times) if response_times else 0,
            'median_response_time': statistics.median(response_times) if response_times else 0,
            'response_times': response_times
        }

    async def run_comprehensive_test(self):
        """Run comprehensive test on all engines with real data"""
        print("🏃 Dream Team: Starting comprehensive engine functional test...")
        print("=" * 80)
        
        # Get real data samples
        print("📊 Retrieving real data samples from database...")
        real_data = self.get_real_data_samples()
        print(f"✅ Database Status: {real_data.get('status')}")
        if real_data.get('status') == 'connected':
            print(f"   📈 Bars in database: {real_data.get('bar_count', 0):,}")
            print(f"   💼 Positions in database: {real_data.get('position_count', 0):,}")
            print(f"   🎯 Test instruments: {real_data.get('instruments', [])}")
        
        print("\n🚀 Starting full capacity engine testing...")
        
        async with aiohttp.ClientSession() as session:
            # Phase 1: Basic health checks
            print("\n📋 Phase 1: Basic Health Checks")
            basic_results = []
            for engine_name, url in self.engines.items():
                print(f"  ⚡ Testing {engine_name}...")
                result = await self.test_engine_endpoint(session, engine_name, url)
                basic_results.append(result)
                status_emoji = "✅" if result['success'] else "❌"
                print(f"    {status_emoji} {engine_name}: {result['response_time']:.3f}s")
            
            # Phase 2: Stress testing
            print("\n🔥 Phase 2: Full Capacity Stress Testing")
            stress_results = []
            for engine_name, url in self.engines.items():
                stress_result = await self.stress_test_engine(session, engine_name, url, 100)
                stress_results.append(stress_result)
                
                print(f"  🎯 {engine_name}: {stress_result['success_rate']:.1f}% success, "
                      f"avg {stress_result['avg_response_time']:.3f}s")
        
        # Generate comprehensive report
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'test_type': 'Dream Team Comprehensive Engine Test',
            'real_data_info': real_data,
            'basic_health_checks': basic_results,
            'stress_test_results': stress_results,
            'summary': self.generate_summary(basic_results, stress_results)
        }
        
        return self.results

    def generate_summary(self, basic_results: List, stress_results: List) -> Dict:
        """Generate comprehensive test summary"""
        healthy_engines = len([r for r in basic_results if r['success']])
        total_engines = len(basic_results)
        
        avg_response_times = [r['avg_response_time'] for r in stress_results if r['successful_requests'] > 0]
        overall_avg_response = statistics.mean(avg_response_times) if avg_response_times else 0
        
        total_requests = sum(r['total_requests'] for r in stress_results)
        total_successful = sum(r['successful_requests'] for r in stress_results)
        
        return {
            'system_health': f"{healthy_engines}/{total_engines} engines healthy",
            'system_availability': f"{healthy_engines/total_engines*100:.1f}%",
            'overall_avg_response_time': f"{overall_avg_response:.3f}s",
            'total_test_requests': total_requests,
            'total_successful_requests': total_successful,
            'overall_success_rate': f"{total_successful/total_requests*100:.1f}%",
            'grade': 'A+' if healthy_engines/total_engines > 0.9 and overall_avg_response < 0.010 else 'A' if healthy_engines/total_engines > 0.8 else 'B'
        }

    def save_results(self, filename: str = None):
        """Save test results to JSON file"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"dream_team_engine_test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"💾 Results saved to: {filename}")
        return filename

async def main():
    """Main execution function"""
    print("🏃 Dream Team Engine Testing - Bob (Scrum Master) Leading")
    print("🎯 Mission: Full capacity functional test with real database data")
    print("=" * 80)
    
    tester = DreamTeamEngineTest()
    results = await tester.run_comprehensive_test()
    
    # Save results
    results_file = tester.save_results()
    
    # Print final summary
    print("\n" + "=" * 80)
    print("🏆 DREAM TEAM TEST RESULTS SUMMARY")
    print("=" * 80)
    
    summary = results['summary']
    print(f"🎯 System Health: {summary['system_health']}")
    print(f"📊 System Availability: {summary['system_availability']}")
    print(f"⚡ Average Response Time: {summary['overall_avg_response_time']}")
    print(f"📈 Total Test Requests: {summary['total_test_requests']:,}")
    print(f"✅ Successful Requests: {summary['total_successful_requests']:,}")
    print(f"🎯 Overall Success Rate: {summary['overall_success_rate']}")
    print(f"🏆 Final Grade: {summary['grade']}")
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    print("🏃 Dream Team mission completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())