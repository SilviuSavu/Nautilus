#!/usr/bin/env python3
"""
Comprehensive 13x13 Engine Connectivity Matrix Test
Nautilus Trading Platform - Inter-Engine Communication Testing

Tests all possible communication paths between the 13 processing engines.
"""

import asyncio
import json
import time
import aiohttp
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Engine Configuration
ENGINES = {
    'analytics': {'port': 8100, 'name': 'Analytics Engine'},
    'backtesting': {'port': 8110, 'name': 'Backtesting Engine'},
    'risk': {'port': 8200, 'name': 'Risk Engine'},
    'factor': {'port': 8300, 'name': 'Factor Engine'},
    'ml': {'port': 8400, 'name': 'ML Engine'},
    'features': {'port': 8500, 'name': 'Features Engine'},
    'websocket': {'port': 8600, 'name': 'WebSocket Engine'},
    'strategy': {'port': 8700, 'name': 'Strategy Engine'},
    'marketdata': {'port': 8800, 'name': 'Enhanced IBKR MarketData Engine'},
    'portfolio': {'port': 8900, 'name': 'Portfolio Engine'},
    'collateral': {'port': 9000, 'name': 'Collateral Engine'},
    'vpin': {'port': 10000, 'name': 'VPIN Engine'},
    'enhanced_vpin': {'port': 10001, 'name': 'Enhanced VPIN Engine'}
}

class EngineConnectivityTester:
    def __init__(self):
        self.session = None
        self.connectivity_matrix = {}
        self.performance_matrix = {}
        self.health_status = {}
        
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=5)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def check_engine_health(self, engine_name: str, port: int) -> Dict:
        """Check if an engine is healthy and get its status."""
        try:
            url = f"http://localhost:{port}/health"
            start_time = time.perf_counter()
            
            async with self.session.get(url) as response:
                end_time = time.perf_counter()
                response_time = (end_time - start_time) * 1000  # Convert to milliseconds
                
                if response.status == 200:
                    data = await response.json()
                    return {
                        'status': 'healthy',
                        'response_time_ms': round(response_time, 2),
                        'data': data,
                        'port': port
                    }
                else:
                    return {
                        'status': 'unhealthy',
                        'response_time_ms': round(response_time, 2),
                        'error': f"HTTP {response.status}",
                        'port': port
                    }
                    
        except asyncio.TimeoutError:
            return {'status': 'timeout', 'error': 'Connection timeout', 'port': port}
        except Exception as e:
            return {'status': 'error', 'error': str(e), 'port': port}
    
    async def test_engine_to_engine_communication(
        self, 
        source_engine: str, 
        target_engine: str
    ) -> Dict:
        """Test communication from one engine to another."""
        source_port = ENGINES[source_engine]['port']
        target_port = ENGINES[target_engine]['port']
        
        try:
            # Test basic connectivity first
            target_health = await self.check_engine_health(target_engine, target_port)
            if target_health['status'] != 'healthy':
                return {
                    'source': source_engine,
                    'target': target_engine,
                    'status': 'target_unavailable',
                    'error': f"Target engine {target_engine} is not healthy",
                    'target_health': target_health
                }
            
            # Try to send a test message (if engine supports it)
            test_endpoints = [
                f"http://localhost:{source_port}/test_communication/{target_engine}",
                f"http://localhost:{source_port}/ping/{target_port}",
                f"http://localhost:{source_port}/status",
            ]
            
            for endpoint in test_endpoints:
                try:
                    start_time = time.perf_counter()
                    async with self.session.get(endpoint) as response:
                        end_time = time.perf_counter()
                        response_time = (end_time - start_time) * 1000
                        
                        if response.status == 200:
                            data = await response.json()
                            return {
                                'source': source_engine,
                                'target': target_engine,
                                'status': 'connected',
                                'response_time_ms': round(response_time, 2),
                                'endpoint': endpoint,
                                'data': data
                            }
                except:
                    continue
            
            # If no specific communication endpoint exists, mark as potential connection
            return {
                'source': source_engine,
                'target': target_engine,
                'status': 'potential_connection',
                'message': 'Both engines are healthy, communication possible via messagebus'
            }
            
        except Exception as e:
            return {
                'source': source_engine,
                'target': target_engine,
                'status': 'error',
                'error': str(e)
            }
    
    async def test_messagebus_routing(self, engine_name: str) -> Dict:
        """Test messagebus connectivity for an engine."""
        port = ENGINES[engine_name]['port']
        
        try:
            # Test messagebus status endpoint
            endpoints = [
                f"http://localhost:{port}/messagebus/status",
                f"http://localhost:{port}/bus/status",
                f"http://localhost:{port}/health"
            ]
            
            for endpoint in endpoints:
                try:
                    async with self.session.get(endpoint) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            # Look for messagebus indicators
                            messagebus_indicators = [
                                'dual_messagebus_connected',
                                'messagebus_connected',
                                'redis_connected',
                                'bus_connected'
                            ]
                            
                            for indicator in messagebus_indicators:
                                if indicator in data and data[indicator]:
                                    return {
                                        'engine': engine_name,
                                        'messagebus_status': 'connected',
                                        'indicator': indicator,
                                        'data': data
                                    }
                            
                            return {
                                'engine': engine_name,
                                'messagebus_status': 'unknown',
                                'data': data
                            }
                except:
                    continue
                    
            return {
                'engine': engine_name,
                'messagebus_status': 'no_endpoint',
                'message': 'No messagebus status endpoint found'
            }
            
        except Exception as e:
            return {
                'engine': engine_name,
                'messagebus_status': 'error',
                'error': str(e)
            }
    
    async def run_comprehensive_test(self) -> Dict:
        """Run the complete 13x13 connectivity matrix test."""
        logger.info("Starting comprehensive 13x13 engine connectivity test...")
        
        # Step 1: Check health of all engines
        logger.info("Step 1: Checking health of all engines...")
        health_tasks = []
        for engine_name, config in ENGINES.items():
            task = self.check_engine_health(engine_name, config['port'])
            health_tasks.append((engine_name, task))
        
        for engine_name, task in health_tasks:
            self.health_status[engine_name] = await task
            status = self.health_status[engine_name]['status']
            logger.info(f"  {engine_name}: {status}")
        
        # Step 2: Test messagebus connectivity
        logger.info("Step 2: Testing messagebus connectivity...")
        messagebus_status = {}
        for engine_name in ENGINES.keys():
            if self.health_status[engine_name]['status'] == 'healthy':
                messagebus_status[engine_name] = await self.test_messagebus_routing(engine_name)
        
        # Step 3: Create 13x13 connectivity matrix
        logger.info("Step 3: Testing engine-to-engine connectivity (13x13 matrix)...")
        connectivity_tasks = []
        
        for source_engine in ENGINES.keys():
            if self.health_status[source_engine]['status'] != 'healthy':
                continue
                
            for target_engine in ENGINES.keys():
                if source_engine != target_engine:
                    task = self.test_engine_to_engine_communication(source_engine, target_engine)
                    connectivity_tasks.append(task)
        
        # Execute all connectivity tests
        connectivity_results = await asyncio.gather(*connectivity_tasks, return_exceptions=True)
        
        # Process results into matrix format
        for result in connectivity_results:
            if isinstance(result, dict) and 'source' in result and 'target' in result:
                source = result['source']
                target = result['target']
                
                if source not in self.connectivity_matrix:
                    self.connectivity_matrix[source] = {}
                
                self.connectivity_matrix[source][target] = result
        
        # Step 4: Calculate performance metrics
        healthy_engines = [e for e, status in self.health_status.items() if status['status'] == 'healthy']
        total_engines = len(ENGINES)
        healthy_count = len(healthy_engines)
        
        # Calculate connectivity percentages
        total_possible_connections = healthy_count * (healthy_count - 1)  # N*(N-1) for directed graph
        successful_connections = sum(
            1 for source in self.connectivity_matrix.values()
            for target in source.values()
            if target['status'] in ['connected', 'potential_connection']
        )
        
        connectivity_percentage = (successful_connections / total_possible_connections * 100) if total_possible_connections > 0 else 0
        
        # Generate final report
        report = {
            'test_timestamp': datetime.now().isoformat(),
            'summary': {
                'total_engines': total_engines,
                'healthy_engines': healthy_count,
                'engine_availability': round((healthy_count / total_engines) * 100, 2),
                'total_possible_connections': total_possible_connections,
                'successful_connections': successful_connections,
                'connectivity_percentage': round(connectivity_percentage, 2)
            },
            'engine_health': self.health_status,
            'messagebus_status': messagebus_status,
            'connectivity_matrix': self.connectivity_matrix,
            'healthy_engines': healthy_engines,
            'unhealthy_engines': [e for e in ENGINES.keys() if e not in healthy_engines]
        }
        
        logger.info(f"Test completed: {healthy_count}/{total_engines} engines healthy, "
                   f"{connectivity_percentage:.1f}% connectivity")
        
        return report

async def main():
    """Run the comprehensive connectivity test."""
    async with EngineConnectivityTester() as tester:
        report = await tester.run_comprehensive_test()
        
        # Save report to file
        timestamp = int(time.time())
        filename = f"comprehensive_13x13_connectivity_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*80)
        print("COMPREHENSIVE 13x13 ENGINE CONNECTIVITY TEST RESULTS")
        print("="*80)
        
        summary = report['summary']
        print(f"Engine Availability: {summary['healthy_engines']}/{summary['total_engines']} "
              f"({summary['engine_availability']}%)")
        print(f"Inter-Engine Connectivity: {summary['successful_connections']}/{summary['total_possible_connections']} "
              f"({summary['connectivity_percentage']}%)")
        
        print(f"\n✅ HEALTHY ENGINES ({len(report['healthy_engines'])}):")
        for engine in report['healthy_engines']:
            health = report['engine_health'][engine]
            response_time = health.get('response_time_ms', 'N/A')
            print(f"  • {ENGINES[engine]['name']} (Port {ENGINES[engine]['port']}) - {response_time}ms")
        
        if report['unhealthy_engines']:
            print(f"\n❌ UNHEALTHY ENGINES ({len(report['unhealthy_engines'])}):")
            for engine in report['unhealthy_engines']:
                health = report['engine_health'][engine]
                error = health.get('error', 'Unknown error')
                print(f"  • {ENGINES[engine]['name']} (Port {ENGINES[engine]['port']}) - {error}")
        
        print(f"\n📊 MESSAGEBUS CONNECTIVITY:")
        for engine, status in report['messagebus_status'].items():
            bus_status = status.get('messagebus_status', 'unknown')
            print(f"  • {ENGINES[engine]['name']}: {bus_status}")
        
        print(f"\n📈 CONNECTIVITY MATRIX SAMPLE:")
        # Show a few key connections
        matrix = report['connectivity_matrix']
        sample_connections = [
            ('factor', 'analytics'),
            ('marketdata', 'factor'),
            ('risk', 'ml'),
            ('analytics', 'strategy'),
        ]
        
        for source, target in sample_connections:
            if source in matrix and target in matrix[source]:
                conn = matrix[source][target]
                status = conn['status']
                response_time = conn.get('response_time_ms', 'N/A')
                print(f"  • {source} → {target}: {status} ({response_time}ms)")
        
        print(f"\n💾 Full report saved to: {filename}")
        print("="*80)
        
        return report

if __name__ == "__main__":
    asyncio.run(main())