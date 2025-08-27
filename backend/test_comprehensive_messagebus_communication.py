#!/usr/bin/env python3
"""
Comprehensive MessageBus Communication Test
Tests direct communication between operational engines via dual messagebus architecture.
"""

import asyncio
import json
import time
import logging
import sys
from typing import Dict, List, Any
import requests

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import dual messagebus components
from dual_messagebus_client import DualMessageBusClient, DualBusConfig, MessageBusType
from universal_enhanced_messagebus_client import EngineType, MessageType, MessagePriority
from ultra_fast_redis_pool import UltraFastRedisPool


class ComprehensiveMessageBusTest:
    """Comprehensive test suite for messagebus communication"""
    
    def __init__(self):
        self.test_results = {
            'engine_health_checks': {},
            'redis_performance_tests': {},
            'direct_communication_tests': {},
            'messagebus_routing_tests': {},
            'latency_measurements': {},
            'throughput_measurements': {},
            'optimization_recommendations': []
        }
        
        # Operational engines
        self.operational_engines = {
            'factor': {'port': 8300, 'name': 'Factor Engine'},
            'marketdata': {'port': 8800, 'name': 'Enhanced IBKR MarketData Engine'},
            'features': {'port': 8500, 'name': 'Features Engine'},  
            'portfolio': {'port': 8900, 'name': 'Portfolio Engine'}
        }
        
        # Dual messagebus clients for testing
        self.test_clients = {}
    
    async def initialize_test_clients(self):
        """Initialize test clients for each engine type"""
        logger.info("🔄 Initializing test clients for communication testing...")
        
        for engine_name, engine_info in self.operational_engines.items():
            try:
                # Map engine names to EngineType
                engine_type_mapping = {
                    'factor': EngineType.FACTOR,
                    'marketdata': EngineType.MARKETDATA,
                    'features': EngineType.FEATURES,
                    'portfolio': EngineType.PORTFOLIO
                }
                
                engine_type = engine_type_mapping.get(engine_name, EngineType.ANALYTICS)
                
                config = DualBusConfig(
                    engine_type=engine_type,
                    engine_instance_id=f"test-{engine_name}-{int(time.time())}"
                )
                
                client = DualMessageBusClient(config)
                await client.initialize()
                
                self.test_clients[engine_name] = client
                logger.info(f"   ✅ {engine_name}: Test client initialized")
                
            except Exception as e:
                logger.error(f"   ❌ {engine_name}: Failed to initialize test client - {str(e)}")
    
    async def test_engine_health(self):
        """Test health of operational engines"""
        logger.info("\n1️⃣ TESTING ENGINE HEALTH STATUS")
        logger.info("=" * 60)
        
        for engine_name, engine_info in self.operational_engines.items():
            port = engine_info['port']
            name = engine_info['name']
            
            try:
                # Test HTTP health endpoint
                start_time = time.time()
                response = requests.get(f"http://localhost:{port}/health", timeout=5)
                response_time = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    health_data = response.json()
                    self.test_results['engine_health_checks'][engine_name] = {
                        'status': 'healthy',
                        'response_time_ms': response_time,
                        'health_data': health_data
                    }
                    
                    # Check for dual messagebus connection
                    dual_bus_connected = health_data.get('dual_messagebus_connected', False)
                    messagebus_status = "✅ Connected" if dual_bus_connected else "❌ Not Connected"
                    
                    logger.info(f"   ✅ {name} (Port {port}): {response_time:.2f}ms - {messagebus_status}")
                    if 'architecture' in health_data:
                        logger.info(f"      Architecture: {health_data['architecture']}")
                    
                else:
                    self.test_results['engine_health_checks'][engine_name] = {
                        'status': 'unhealthy',
                        'response_time_ms': response_time,
                        'http_status': response.status_code
                    }
                    logger.warning(f"   ⚠️ {name} (Port {port}): HTTP {response.status_code}")
                    
            except Exception as e:
                self.test_results['engine_health_checks'][engine_name] = {
                    'status': 'error',
                    'error': str(e)
                }
                logger.error(f"   ❌ {name} (Port {port}): {str(e)}")
    
    async def test_redis_pool_performance(self):
        """Test Redis pool performance across all instances"""
        logger.info("\n2️⃣ TESTING ULTRA-FAST REDIS POOL PERFORMANCE")
        logger.info("=" * 60)
        
        pool_manager = UltraFastRedisPool()
        
        # Test MarketData Bus (6380)
        logger.info("📡 Testing MarketData Bus (Port 6380)...")
        marketdata_client = pool_manager.get_marketdata_pool()
        marketdata_results = await pool_manager.test_pool_performance('marketdata', 1000)
        self.test_results['redis_performance_tests']['marketdata_bus'] = marketdata_results
        
        # Test Engine Logic Bus (6381)
        logger.info("⚙️ Testing Engine Logic Bus (Port 6381)...")
        engine_logic_client = pool_manager.get_engine_logic_pool()
        engine_logic_results = await pool_manager.test_pool_performance('engine_logic', 1000)
        self.test_results['redis_performance_tests']['engine_logic_bus'] = engine_logic_results
        
        # Test Primary Redis (6379) - if available
        try:
            from ultra_fast_redis_pool import PoolConfiguration
            primary_config = PoolConfiguration(host='localhost', port=6379)
            pool_manager.create_optimized_pool('primary', primary_config)
            primary_results = await pool_manager.test_pool_performance('primary', 1000)
            self.test_results['redis_performance_tests']['primary_redis'] = primary_results
            logger.info("🔧 Testing Primary Redis (Port 6379)...")
        except Exception as e:
            logger.debug(f"Primary Redis test skipped: {e}")
        
        # Test Neural GPU Bus (6382) - if available
        try:
            from ultra_fast_redis_pool import PoolConfiguration
            neural_config = PoolConfiguration(host='localhost', port=6382)
            pool_manager.create_optimized_pool('neural_gpu', neural_config)
            neural_results = await pool_manager.test_pool_performance('neural_gpu', 1000)
            self.test_results['redis_performance_tests']['neural_gpu_bus'] = neural_results
            logger.info("🧠 Testing Neural GPU Bus (Port 6382)...")
        except Exception as e:
            logger.debug(f"Neural GPU Bus test skipped: {e}")
        
        # Close pool manager
        await pool_manager.close_all_pools()
    
    async def test_direct_engine_communication(self):
        """Test direct communication between engines"""
        logger.info("\n3️⃣ TESTING DIRECT ENGINE COMMUNICATION")
        logger.info("=" * 60)
        
        # Test Factor Engine (8300) -> MarketData Engine (8800) communication
        if 'factor' in self.test_clients and 'marketdata' in self.test_clients:
            logger.info("🔄 Testing Factor Engine -> MarketData Engine communication...")
            
            factor_client = self.test_clients['factor']
            marketdata_client = self.test_clients['marketdata']
            
            try:
                # Send test message from Factor to MarketData
                test_payload = {
                    'test_type': 'direct_communication',
                    'source': 'factor_engine',
                    'target': 'marketdata_engine',
                    'timestamp': time.time(),
                    'test_data': {'symbol': 'AAPL', 'factor_value': 0.85}
                }
                
                start_time = time.time()
                success = await factor_client.publish_message(
                    MessageType.FACTOR_CALCULATION,
                    test_payload,
                    MessagePriority.HIGH
                )
                send_latency = (time.time() - start_time) * 1000
                
                self.test_results['direct_communication_tests']['factor_to_marketdata'] = {
                    'success': success,
                    'send_latency_ms': send_latency,
                    'message_type': 'FACTOR_CALCULATION',
                    'bus_used': 'engine_logic_bus'
                }
                
                if success:
                    logger.info(f"   ✅ Factor -> MarketData: Message sent successfully ({send_latency:.3f}ms)")
                else:
                    logger.error(f"   ❌ Factor -> MarketData: Message send failed")
                    
            except Exception as e:
                logger.error(f"   ❌ Factor -> MarketData communication error: {e}")
                self.test_results['direct_communication_tests']['factor_to_marketdata'] = {
                    'success': False,
                    'error': str(e)
                }
        
        # Test MarketData Engine -> All Engines broadcast
        if 'marketdata' in self.test_clients:
            logger.info("📡 Testing MarketData Engine broadcast to all engines...")
            
            marketdata_client = self.test_clients['marketdata']
            
            try:
                # Broadcast market data update
                market_update = {
                    'symbol': 'SPY',
                    'price': 450.75,
                    'volume': 1000000,
                    'timestamp': time.time(),
                    'source': 'ibkr_enhanced'
                }
                
                start_time = time.time()
                success = await marketdata_client.publish_message(
                    MessageType.MARKET_DATA,
                    market_update,
                    MessagePriority.HIGH
                )
                broadcast_latency = (time.time() - start_time) * 1000
                
                self.test_results['direct_communication_tests']['marketdata_broadcast'] = {
                    'success': success,
                    'broadcast_latency_ms': broadcast_latency,
                    'message_type': 'MARKET_DATA',
                    'bus_used': 'marketdata_bus'
                }
                
                if success:
                    logger.info(f"   ✅ MarketData broadcast: Sent successfully ({broadcast_latency:.3f}ms)")
                else:
                    logger.error(f"   ❌ MarketData broadcast: Send failed")
                    
            except Exception as e:
                logger.error(f"   ❌ MarketData broadcast error: {e}")
                self.test_results['direct_communication_tests']['marketdata_broadcast'] = {
                    'success': False,
                    'error': str(e)
                }
    
    async def test_messagebus_routing(self):
        """Test message routing between different buses"""
        logger.info("\n4️⃣ TESTING MESSAGEBUS ROUTING VERIFICATION")
        logger.info("=" * 60)
        
        if not self.test_clients:
            logger.warning("No test clients available for routing tests")
            return
        
        # Get a test client
        test_client_name = next(iter(self.test_clients.keys()))
        test_client = self.test_clients[test_client_name]
        
        # Test MarketData Bus routing
        logger.info("📡 Testing MarketData Bus routing...")
        marketdata_messages = [
            (MessageType.MARKET_DATA, {'symbol': 'AAPL', 'price': 180.50}),
            (MessageType.PRICE_UPDATE, {'symbol': 'MSFT', 'price': 420.75}),
            (MessageType.TRADE_EXECUTION, {'symbol': 'GOOGL', 'volume': 1000})
        ]
        
        marketdata_routing_results = []
        for message_type, payload in marketdata_messages:
            try:
                start_time = time.time()
                success = await test_client.publish_message(message_type, payload)
                latency = (time.time() - start_time) * 1000
                
                marketdata_routing_results.append({
                    'message_type': message_type.value,
                    'success': success,
                    'latency_ms': latency,
                    'expected_bus': 'marketdata_bus'
                })
                
                if success:
                    logger.info(f"   ✅ {message_type.value}: Routed to MarketData Bus ({latency:.3f}ms)")
                else:
                    logger.error(f"   ❌ {message_type.value}: Routing failed")
                    
            except Exception as e:
                logger.error(f"   ❌ {message_type.value}: Error - {e}")
                marketdata_routing_results.append({
                    'message_type': message_type.value,
                    'success': False,
                    'error': str(e)
                })
        
        self.test_results['messagebus_routing_tests']['marketdata_bus'] = marketdata_routing_results
        
        # Test Engine Logic Bus routing
        logger.info("⚙️ Testing Engine Logic Bus routing...")
        engine_logic_messages = [
            (MessageType.RISK_METRIC, {'symbol': 'SPY', 'var': 0.025}),
            (MessageType.ML_PREDICTION, {'symbol': 'QQQ', 'prediction': 0.75}),
            (MessageType.STRATEGY_SIGNAL, {'action': 'BUY', 'symbol': 'IWM'}),
            (MessageType.ANALYTICS_RESULT, {'metric': 'volatility', 'value': 0.18})
        ]
        
        engine_logic_routing_results = []
        for message_type, payload in engine_logic_messages:
            try:
                start_time = time.time()
                success = await test_client.publish_message(message_type, payload)
                latency = (time.time() - start_time) * 1000
                
                engine_logic_routing_results.append({
                    'message_type': message_type.value,
                    'success': success,
                    'latency_ms': latency,
                    'expected_bus': 'engine_logic_bus'
                })
                
                if success:
                    logger.info(f"   ✅ {message_type.value}: Routed to Engine Logic Bus ({latency:.3f}ms)")
                else:
                    logger.error(f"   ❌ {message_type.value}: Routing failed")
                    
            except Exception as e:
                logger.error(f"   ❌ {message_type.value}: Error - {e}")
                engine_logic_routing_results.append({
                    'message_type': message_type.value,
                    'success': False,
                    'error': str(e)
                })
        
        self.test_results['messagebus_routing_tests']['engine_logic_bus'] = engine_logic_routing_results
    
    async def measure_latency_and_throughput(self):
        """Measure comprehensive latency and throughput metrics"""
        logger.info("\n5️⃣ MEASURING LATENCY AND THROUGHPUT METRICS")
        logger.info("=" * 60)
        
        if not self.test_clients:
            logger.warning("No test clients available for performance measurements")
            return
        
        # Get test client for performance measurements
        test_client_name = next(iter(self.test_clients.keys()))
        test_client = self.test_clients[test_client_name]
        
        # High-frequency message throughput test
        logger.info("🚀 Testing high-frequency message throughput...")
        
        total_messages = 1000
        successful_messages = 0
        failed_messages = 0
        latencies = []
        
        start_time = time.time()
        
        for i in range(total_messages):
            try:
                message_start = time.time()
                
                # Alternate between message types to test both buses
                if i % 2 == 0:
                    # MarketData Bus message
                    success = await test_client.publish_message(
                        MessageType.MARKET_DATA,
                        {'test_id': i, 'symbol': 'TEST', 'price': 100.0 + i * 0.01}
                    )
                else:
                    # Engine Logic Bus message
                    success = await test_client.publish_message(
                        MessageType.ANALYTICS_RESULT,
                        {'test_id': i, 'metric': 'test_metric', 'value': i * 0.1}
                    )
                
                message_latency = (time.time() - message_start) * 1000
                latencies.append(message_latency)
                
                if success:
                    successful_messages += 1
                else:
                    failed_messages += 1
                    
            except Exception as e:
                failed_messages += 1
                logger.debug(f"Message {i} failed: {e}")
        
        total_duration = time.time() - start_time
        
        # Calculate metrics
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
        else:
            avg_latency = min_latency = max_latency = p95_latency = 0
        
        throughput = successful_messages / total_duration if total_duration > 0 else 0
        success_rate = successful_messages / total_messages if total_messages > 0 else 0
        
        self.test_results['latency_measurements'] = {
            'avg_latency_ms': avg_latency,
            'min_latency_ms': min_latency,
            'max_latency_ms': max_latency,
            'p95_latency_ms': p95_latency
        }
        
        self.test_results['throughput_measurements'] = {
            'total_messages': total_messages,
            'successful_messages': successful_messages,
            'failed_messages': failed_messages,
            'throughput_msgs_per_sec': throughput,
            'success_rate': success_rate,
            'total_duration_sec': total_duration
        }
        
        logger.info(f"   📊 Average Latency: {avg_latency:.3f}ms")
        logger.info(f"   📊 P95 Latency: {p95_latency:.3f}ms")
        logger.info(f"   📊 Throughput: {throughput:.0f} messages/sec")
        logger.info(f"   📊 Success Rate: {success_rate * 100:.1f}%")
    
    def generate_optimization_recommendations(self):
        """Generate performance optimization recommendations"""
        logger.info("\n6️⃣ GENERATING OPTIMIZATION RECOMMENDATIONS")
        logger.info("=" * 60)
        
        recommendations = []
        
        # Analyze Redis performance
        if 'redis_performance_tests' in self.test_results:
            redis_tests = self.test_results['redis_performance_tests']
            
            # Check latency performance
            for bus_name, results in redis_tests.items():
                avg_latency = results.get('avg_latency_ms', 0)
                throughput = results.get('throughput_ops_sec', 0)
                
                if avg_latency > 1.0:  # > 1ms
                    recommendations.append({
                        'category': 'Redis Performance',
                        'priority': 'HIGH',
                        'issue': f'{bus_name} average latency is {avg_latency:.3f}ms (target: <1ms)',
                        'recommendation': f'Optimize {bus_name} socket timeout and connection pooling settings'
                    })
                elif avg_latency > 0.5:  # > 0.5ms
                    recommendations.append({
                        'category': 'Redis Performance',
                        'priority': 'MEDIUM',
                        'issue': f'{bus_name} average latency is {avg_latency:.3f}ms (target: <0.5ms)',
                        'recommendation': f'Fine-tune {bus_name} TCP keepalive and buffer settings'
                    })
                
                if throughput < 5000:  # < 5000 ops/sec
                    recommendations.append({
                        'category': 'Redis Throughput',
                        'priority': 'HIGH',
                        'issue': f'{bus_name} throughput is {throughput:.0f} ops/sec (target: >5000)',
                        'recommendation': f'Increase {bus_name} max connections and optimize connection reuse'
                    })
        
        # Analyze engine health
        if 'engine_health_checks' in self.test_results:
            health_checks = self.test_results['engine_health_checks']
            
            for engine_name, health_data in health_checks.items():
                if health_data.get('status') != 'healthy':
                    recommendations.append({
                        'category': 'Engine Health',
                        'priority': 'CRITICAL',
                        'issue': f'{engine_name} engine is not responding properly',
                        'recommendation': f'Investigate {engine_name} engine startup and dependency issues'
                    })
                
                # Check for messagebus connectivity
                if 'health_data' in health_data:
                    dual_bus_connected = health_data['health_data'].get('dual_messagebus_connected', False)
                    if not dual_bus_connected:
                        recommendations.append({
                            'category': 'MessageBus Integration',
                            'priority': 'HIGH',
                            'issue': f'{engine_name} engine not connected to dual messagebus',
                            'recommendation': f'Update {engine_name} engine to use DualMessageBusClient'
                        })
        
        # Analyze communication tests
        if 'direct_communication_tests' in self.test_results:
            comm_tests = self.test_results['direct_communication_tests']
            
            for test_name, test_data in comm_tests.items():
                if not test_data.get('success', False):
                    recommendations.append({
                        'category': 'Engine Communication',
                        'priority': 'HIGH',
                        'issue': f'Direct communication test {test_name} failed',
                        'recommendation': f'Debug messagebus routing and stream configuration for {test_name}'
                    })
        
        # Analyze throughput performance
        if 'throughput_measurements' in self.test_results:
            throughput_data = self.test_results['throughput_measurements']
            throughput = throughput_data.get('throughput_msgs_per_sec', 0)
            success_rate = throughput_data.get('success_rate', 0)
            
            if throughput < 500:  # < 500 msgs/sec
                recommendations.append({
                    'category': 'System Throughput',
                    'priority': 'HIGH',
                    'issue': f'System throughput is {throughput:.0f} msgs/sec (target: >1000)',
                    'recommendation': 'Optimize messagebus configuration and increase connection pool sizes'
                })
            
            if success_rate < 0.95:  # < 95% success rate
                recommendations.append({
                    'category': 'Message Reliability',
                    'priority': 'CRITICAL',
                    'issue': f'Message success rate is {success_rate * 100:.1f}% (target: >95%)',
                    'recommendation': 'Investigate message failures and implement retry mechanisms'
                })
        
        # General recommendations
        recommendations.extend([
            {
                'category': 'Performance Optimization',
                'priority': 'MEDIUM',
                'issue': 'Standard optimization opportunity',
                'recommendation': 'Enable Redis pipelining for batch operations'
            },
            {
                'category': 'Monitoring',
                'priority': 'MEDIUM',
                'issue': 'System observability',
                'recommendation': 'Implement comprehensive messagebus metrics collection'
            },
            {
                'category': 'Architecture',
                'priority': 'LOW',
                'issue': 'Future scalability',
                'recommendation': 'Consider Redis Cluster for horizontal scaling when traffic increases'
            }
        ])
        
        self.test_results['optimization_recommendations'] = recommendations
        
        # Display recommendations by priority
        for priority in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            priority_recs = [r for r in recommendations if r['priority'] == priority]
            if priority_recs:
                logger.info(f"\n{priority} Priority Recommendations:")
                for i, rec in enumerate(priority_recs, 1):
                    logger.info(f"   {i}. [{rec['category']}] {rec['recommendation']}")
                    logger.info(f"      Issue: {rec['issue']}")
    
    async def close_test_clients(self):
        """Close all test clients"""
        logger.info("\n🔄 Closing test clients...")
        
        for engine_name, client in self.test_clients.items():
            try:
                await client.close()
                logger.info(f"   ✅ {engine_name}: Test client closed")
            except Exception as e:
                logger.error(f"   ❌ {engine_name}: Error closing test client - {e}")
    
    def print_comprehensive_summary(self):
        """Print comprehensive test results summary"""
        logger.info("\n" + "="*80)
        logger.info("🎯 COMPREHENSIVE MESSAGEBUS PERFORMANCE REPORT")
        logger.info("="*80)
        
        # Engine Health Summary
        logger.info("\n📊 ENGINE HEALTH SUMMARY:")
        healthy_engines = 0
        total_engines = len(self.operational_engines)
        
        for engine_name, health_data in self.test_results.get('engine_health_checks', {}).items():
            status = health_data.get('status', 'unknown')
            if status == 'healthy':
                healthy_engines += 1
                response_time = health_data.get('response_time_ms', 0)
                dual_bus = health_data.get('health_data', {}).get('dual_messagebus_connected', False)
                bus_status = "✅ Dual Bus" if dual_bus else "❌ No Dual Bus"
                logger.info(f"   ✅ {engine_name}: {response_time:.2f}ms - {bus_status}")
            else:
                logger.info(f"   ❌ {engine_name}: {status}")
        
        logger.info(f"\nEngine Health: {healthy_engines}/{total_engines} engines operational ({healthy_engines/total_engines*100:.1f}%)")
        
        # Redis Performance Summary
        logger.info("\n📊 REDIS PERFORMANCE SUMMARY:")
        for bus_name, results in self.test_results.get('redis_performance_tests', {}).items():
            avg_latency = results.get('avg_latency_ms', 0)
            throughput = results.get('throughput_ops_sec', 0)
            success_rate = results.get('success_rate', 0)
            
            logger.info(f"   {bus_name.replace('_', ' ').title()}:")
            logger.info(f"     Latency: {avg_latency:.3f}ms | Throughput: {throughput:.0f} ops/sec | Success: {success_rate*100:.1f}%")
        
        # Communication Test Summary
        logger.info("\n📊 COMMUNICATION TEST SUMMARY:")
        comm_tests = self.test_results.get('direct_communication_tests', {})
        successful_tests = sum(1 for test in comm_tests.values() if test.get('success', False))
        total_tests = len(comm_tests)
        
        if total_tests > 0:
            logger.info(f"   Direct Communication: {successful_tests}/{total_tests} tests passed ({successful_tests/total_tests*100:.1f}%)")
            
            for test_name, test_data in comm_tests.items():
                if test_data.get('success'):
                    latency = test_data.get('send_latency_ms', 0) or test_data.get('broadcast_latency_ms', 0)
                    logger.info(f"     ✅ {test_name}: {latency:.3f}ms")
                else:
                    logger.info(f"     ❌ {test_name}: Failed")
        
        # System Performance Summary
        logger.info("\n📊 SYSTEM PERFORMANCE SUMMARY:")
        if 'latency_measurements' in self.test_results:
            latency_data = self.test_results['latency_measurements']
            logger.info(f"   Average Latency: {latency_data.get('avg_latency_ms', 0):.3f}ms")
            logger.info(f"   P95 Latency: {latency_data.get('p95_latency_ms', 0):.3f}ms")
        
        if 'throughput_measurements' in self.test_results:
            throughput_data = self.test_results['throughput_measurements']
            logger.info(f"   System Throughput: {throughput_data.get('throughput_msgs_per_sec', 0):.0f} messages/sec")
            logger.info(f"   Message Success Rate: {throughput_data.get('success_rate', 0)*100:.1f}%")
        
        # Optimization Summary
        recommendations = self.test_results.get('optimization_recommendations', [])
        critical_recs = [r for r in recommendations if r['priority'] == 'CRITICAL']
        high_recs = [r for r in recommendations if r['priority'] == 'HIGH']
        
        logger.info(f"\n📊 OPTIMIZATION RECOMMENDATIONS: {len(recommendations)} total")
        if critical_recs:
            logger.info(f"   🚨 CRITICAL: {len(critical_recs)} issues require immediate attention")
        if high_recs:
            logger.info(f"   ⚠️ HIGH: {len(high_recs)} issues should be addressed soon")
        
        logger.info("\n" + "="*80)


async def main():
    """Main test execution"""
    logger.info("🚀 COMPREHENSIVE MESSAGEBUS COMMUNICATION TEST")
    logger.info("Testing dual messagebus architecture with operational engines")
    logger.info("="*80)
    
    test_suite = ComprehensiveMessageBusTest()
    
    try:
        # Initialize test clients
        await test_suite.initialize_test_clients()
        
        # Run comprehensive tests
        await test_suite.test_engine_health()
        await test_suite.test_redis_pool_performance()
        await test_suite.test_direct_engine_communication()
        await test_suite.test_messagebus_routing()
        await test_suite.measure_latency_and_throughput()
        
        # Generate recommendations
        test_suite.generate_optimization_recommendations()
        
        # Print comprehensive summary
        test_suite.print_comprehensive_summary()
        
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
    
    finally:
        # Clean up
        await test_suite.close_test_clients()
        
        # Write results to file
        import json
        timestamp = int(time.time())
        results_file = f"comprehensive_messagebus_test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(test_suite.test_results, f, indent=2, default=str)
        
        logger.info(f"\n📄 Test results saved to: {results_file}")


if __name__ == "__main__":
    asyncio.run(main())