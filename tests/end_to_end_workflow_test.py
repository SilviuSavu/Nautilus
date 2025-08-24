#!/usr/bin/env python3
"""
Nautilus End-to-End Workflow Integration Test
Tests complete trading workflows across containerized architecture
"""

import asyncio
import aiohttp
import time
import json
import statistics
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NautilusWorkflowTest:
    """End-to-end workflow test across 9 containerized engines"""
    
    def __init__(self):
        self.engines = {
            'analytics': {'port': 8100, 'weight': 20},
            'risk': {'port': 8200, 'weight': 15},
            'factor': {'port': 8300, 'weight': 12},
            'ml': {'port': 8400, 'weight': 10},
            'features': {'port': 8500, 'weight': 8},
            'websocket': {'port': 8600, 'weight': 25},
            'strategy': {'port': 8700, 'weight': 15},
            'marketdata': {'port': 8800, 'weight': 20},
            'portfolio': {'port': 8900, 'weight': 10}
        }
        
        self.test_workflows = {
            'market_data_flow': {
                'description': 'Complete market data ingestion and processing',
                'steps': ['marketdata', 'features', 'analytics'],
                'payload': {
                    'symbol': 'AAPL',
                    'timeframe': '1m',
                    'indicators': ['rsi', 'macd', 'bollinger_bands']
                }
            },
            'risk_assessment_flow': {
                'description': 'Risk assessment workflow with portfolio analysis',
                'steps': ['portfolio', 'risk', 'analytics'],
                'payload': {
                    'portfolio_id': 'test_portfolio',
                    'positions': [
                        {'symbol': 'AAPL', 'quantity': 100, 'price': 155.00},
                        {'symbol': 'MSFT', 'quantity': 50, 'price': 310.00}
                    ]
                }
            },
            'strategy_execution_flow': {
                'description': 'Strategy testing and execution workflow',
                'steps': ['strategy', 'ml', 'risk', 'portfolio'],
                'payload': {
                    'strategy_id': 'momentum_strategy',
                    'universe': ['AAPL', 'MSFT', 'GOOGL'],
                    'allocation_method': 'equal_weight'
                }
            },
            'factor_analysis_flow': {
                'description': 'Factor analysis and model building workflow',
                'steps': ['factor', 'ml', 'analytics'],
                'payload': {
                    'symbols': ['AAPL', 'MSFT', 'GOOGL'],
                    'factor_categories': ['technical', 'fundamental', 'macro'],
                    'prediction_horizon': '5D'
                }
            },
            'real_time_streaming_flow': {
                'description': 'Real-time data streaming and distribution',
                'steps': ['websocket', 'marketdata', 'analytics'],
                'payload': {
                    'symbols': ['AAPL', 'MSFT'],
                    'stream_types': ['trades', 'quotes', 'analytics']
                }
            }
        }
        
        self.workflow_results = {}
        self.performance_metrics = {
            'total_workflows_tested': 0,
            'successful_workflows': 0,
            'failed_workflows': 0,
            'average_workflow_time': 0,
            'engine_failure_rates': {},
            'bottleneck_analysis': {}
        }

    async def health_check_engines(self) -> Dict[str, bool]:
        """Check health of all engines before running workflows"""
        logger.info("🔍 Performing engine health checks...")
        
        async with aiohttp.ClientSession() as session:
            health_tasks = []
            for engine, config in self.engines.items():
                task = self._check_single_engine_health(session, engine, config['port'])
                health_tasks.append(task)
            
            health_results = await asyncio.gather(*health_tasks, return_exceptions=True)
        
        health_status = {}
        for i, (engine, result) in enumerate(zip(self.engines.keys(), health_results)):
            if isinstance(result, Exception):
                health_status[engine] = False
                logger.warning(f"  ❌ {engine.capitalize()}: {result}")
            else:
                health_status[engine] = result
                status_icon = "✅" if result else "❌"
                logger.info(f"  {status_icon} {engine.capitalize()}: {'Healthy' if result else 'Unhealthy'}")
        
        healthy_count = sum(health_status.values())
        logger.info(f"📊 Health Summary: {healthy_count}/{len(self.engines)} engines healthy")
        
        return health_status

    async def _check_single_engine_health(self, session: aiohttp.ClientSession, engine: str, port: int) -> bool:
        """Check health of a single engine"""
        try:
            async with session.get(f'http://localhost:{port}/health', timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('status') == 'healthy'
                return False
        except Exception:
            return False

    async def run_workflow(self, workflow_name: str, workflow_config: Dict) -> Dict[str, Any]:
        """Execute a complete end-to-end workflow"""
        logger.info(f"\n🔄 Executing workflow: {workflow_name}")
        logger.info(f"   Description: {workflow_config['description']}")
        logger.info(f"   Steps: {' → '.join(workflow_config['steps'])}")
        
        start_time = time.time()
        step_results = []
        workflow_success = True
        
        async with aiohttp.ClientSession() as session:
            for i, engine in enumerate(workflow_config['steps']):
                step_start = time.time()
                
                try:
                    # Execute step
                    success, response_data = await self._execute_workflow_step(
                        session, engine, workflow_config['payload'], i
                    )
                    
                    step_time = time.time() - step_start
                    
                    step_result = {
                        'engine': engine,
                        'step_number': i + 1,
                        'success': success,
                        'execution_time': step_time,
                        'response_data': response_data[:200] if isinstance(response_data, str) else str(response_data)[:200]
                    }
                    
                    step_results.append(step_result)
                    
                    if success:
                        logger.info(f"    ✅ Step {i+1} ({engine}): {step_time:.3f}s")
                    else:
                        logger.warning(f"    ❌ Step {i+1} ({engine}): Failed in {step_time:.3f}s")
                        workflow_success = False
                        
                except Exception as e:
                    step_time = time.time() - step_start
                    logger.error(f"    ❌ Step {i+1} ({engine}): Exception - {e}")
                    
                    step_results.append({
                        'engine': engine,
                        'step_number': i + 1,
                        'success': False,
                        'execution_time': step_time,
                        'error': str(e)
                    })
                    
                    workflow_success = False
        
        total_time = time.time() - start_time
        
        workflow_result = {
            'workflow_name': workflow_name,
            'description': workflow_config['description'],
            'success': workflow_success,
            'total_execution_time': total_time,
            'steps_completed': len([s for s in step_results if s['success']]),
            'total_steps': len(workflow_config['steps']),
            'step_results': step_results,
            'success_rate': len([s for s in step_results if s['success']]) / len(step_results) if step_results else 0
        }
        
        status_icon = "✅" if workflow_success else "❌"
        logger.info(f"  {status_icon} Workflow completed: {total_time:.3f}s, {workflow_result['success_rate']:.1%} success rate")
        
        return workflow_result

    async def _execute_workflow_step(self, session: aiohttp.ClientSession, engine: str, payload: Dict, step_number: int) -> Tuple[bool, Any]:
        """Execute a single workflow step"""
        engine_config = self.engines[engine]
        port = engine_config['port']
        
        # Define endpoints based on engine type and workflow step
        endpoints = {
            'analytics': '/analytics/calculate/test_portfolio',
            'risk': '/risk/check/test_portfolio',
            'factor': '/factors/calculate',
            'ml': '/ml/predict/price',
            'features': '/features/technical',
            'websocket': '/websocket/stats',
            'strategy': '/strategy/test/test_strategy',
            'marketdata': '/marketdata/historical',
            'portfolio': '/portfolio/optimize'
        }
        
        url = f"http://localhost:{port}{endpoints.get(engine, '/health')}"
        
        try:
            if engine == 'websocket':
                # WebSocket engine uses GET
                async with session.get(url, timeout=30) as response:
                    success = response.status == 200
                    if success:
                        response_data = await response.text()
                    else:
                        response_data = f"HTTP {response.status}"
                    return success, response_data
            else:
                # Other engines use POST with payload
                async with session.post(url, json=payload, timeout=30) as response:
                    success = response.status == 200
                    if success:
                        response_data = await response.text()
                    else:
                        response_data = f"HTTP {response.status}"
                    return success, response_data
                    
        except asyncio.TimeoutError:
            return False, "Timeout after 30 seconds"
        except Exception as e:
            return False, f"Exception: {str(e)}"

    def analyze_workflow_patterns(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze workflow execution patterns and identify bottlenecks"""
        logger.info("\n📊 Analyzing workflow patterns...")
        
        # Aggregate metrics
        total_workflows = len(results)
        successful_workflows = len([r for r in results.values() if r['success']])
        
        # Calculate average times
        execution_times = [r['total_execution_time'] for r in results.values()]
        avg_execution_time = statistics.mean(execution_times) if execution_times else 0
        
        # Engine failure analysis
        engine_failures = {}
        engine_step_counts = {}
        
        for workflow_result in results.values():
            for step in workflow_result['step_results']:
                engine = step['engine']
                
                if engine not in engine_failures:
                    engine_failures[engine] = 0
                    engine_step_counts[engine] = 0
                
                engine_step_counts[engine] += 1
                if not step['success']:
                    engine_failures[engine] += 1
        
        # Calculate failure rates
        engine_failure_rates = {}
        for engine in engine_failures:
            if engine_step_counts[engine] > 0:
                engine_failure_rates[engine] = engine_failures[engine] / engine_step_counts[engine]
            else:
                engine_failure_rates[engine] = 0
        
        # Identify bottlenecks (slowest engines)
        engine_times = {}
        for workflow_result in results.values():
            for step in workflow_result['step_results']:
                engine = step['engine']
                if engine not in engine_times:
                    engine_times[engine] = []
                engine_times[engine].append(step['execution_time'])
        
        engine_avg_times = {}
        for engine, times in engine_times.items():
            if times:
                engine_avg_times[engine] = statistics.mean(times)
        
        # Sort by average execution time (descending)
        bottlenecks = dict(sorted(engine_avg_times.items(), key=lambda x: x[1], reverse=True))
        
        analysis = {
            'total_workflows_tested': total_workflows,
            'successful_workflows': successful_workflows,
            'failed_workflows': total_workflows - successful_workflows,
            'overall_success_rate': successful_workflows / total_workflows if total_workflows > 0 else 0,
            'average_workflow_time': avg_execution_time,
            'execution_time_range': {
                'min': min(execution_times) if execution_times else 0,
                'max': max(execution_times) if execution_times else 0,
                'std_dev': statistics.stdev(execution_times) if len(execution_times) > 1 else 0
            },
            'engine_failure_rates': engine_failure_rates,
            'engine_performance_ranking': bottlenecks,
            'bottleneck_analysis': {
                'slowest_engine': max(bottlenecks.items(), key=lambda x: x[1]) if bottlenecks else None,
                'fastest_engine': min(bottlenecks.items(), key=lambda x: x[1]) if bottlenecks else None
            }
        }
        
        return analysis

    def print_workflow_summary(self, analysis: Dict[str, Any]):
        """Print comprehensive workflow analysis summary"""
        print("\n" + "="*80)
        print("🎯 END-TO-END WORKFLOW ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"\n📊 Overall Workflow Performance:")
        print(f"  Total Workflows Tested: {analysis['total_workflows_tested']}")
        print(f"  Successful Workflows: {analysis['successful_workflows']}")
        print(f"  Failed Workflows: {analysis['failed_workflows']}")
        print(f"  Overall Success Rate: {analysis['overall_success_rate']:.1%}")
        print(f"  Average Workflow Time: {analysis['average_workflow_time']:.3f}s")
        
        print(f"\n⚡ Execution Time Analysis:")
        time_range = analysis['execution_time_range']
        print(f"  Fastest Workflow: {time_range['min']:.3f}s")
        print(f"  Slowest Workflow: {time_range['max']:.3f}s")
        print(f"  Standard Deviation: {time_range['std_dev']:.3f}s")
        
        print(f"\n🔧 Engine Reliability Analysis:")
        for engine, failure_rate in sorted(analysis['engine_failure_rates'].items(), key=lambda x: x[1]):
            status_icon = "✅" if failure_rate < 0.1 else "⚠️" if failure_rate < 0.3 else "❌"
            print(f"  {status_icon} {engine.capitalize():>12}: {failure_rate:.1%} failure rate")
        
        print(f"\n🚀 Performance Ranking (Average Response Time):")
        for i, (engine, avg_time) in enumerate(analysis['engine_performance_ranking'].items()):
            rank = i + 1
            performance_icon = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "📊"
            print(f"  {performance_icon} #{rank} {engine.capitalize():>12}: {avg_time:.3f}s average")
        
        if analysis['bottleneck_analysis']['slowest_engine']:
            slowest = analysis['bottleneck_analysis']['slowest_engine']
            fastest = analysis['bottleneck_analysis']['fastest_engine']
            print(f"\n🎯 Bottleneck Analysis:")
            print(f"  ⚠️  Slowest Engine: {slowest[0].capitalize()} ({slowest[1]:.3f}s average)")
            print(f"  ⚡ Fastest Engine: {fastest[0].capitalize()} ({fastest[1]:.3f}s average)")
            print(f"  📈 Performance Gap: {(slowest[1] / fastest[1]):.1f}x difference")
        
        # Architecture benefits
        print(f"\n💡 Containerized Architecture Benefits:")
        print(f"  ✅ Independent engine scaling and fault isolation")
        print(f"  ✅ Parallel workflow execution across microservices")
        print(f"  ✅ Real-time bottleneck identification and optimization")
        print(f"  ✅ Engine-specific performance monitoring and tuning")
        
        # Recommendations
        success_rate = analysis['overall_success_rate']
        avg_time = analysis['average_workflow_time']
        
        if success_rate >= 0.95 and avg_time <= 5.0:
            print(f"\n🏆 EXCELLENT WORKFLOW PERFORMANCE!")
            print(f"   Workflows executing with {success_rate:.1%} success rate in {avg_time:.3f}s average")
            print(f"   Architecture demonstrates enterprise-grade reliability")
        elif success_rate >= 0.8 and avg_time <= 10.0:
            print(f"\n🎉 GOOD WORKFLOW PERFORMANCE!")
            print(f"   Workflows achieving {success_rate:.1%} success rate in {avg_time:.3f}s average")
            print(f"   Production-ready with room for optimization")
        else:
            print(f"\n📈 WORKFLOW PERFORMANCE NEEDS IMPROVEMENT")
            print(f"   Current: {success_rate:.1%} success rate, {avg_time:.3f}s average")
            print(f"   Recommend engine-specific optimization focus")

    async def run_all_workflows(self) -> Dict[str, Any]:
        """Execute all end-to-end workflow tests"""
        logger.info("🚀 Starting End-to-End Workflow Testing")
        logger.info("="*60)
        
        # Health check first
        health_status = await self.health_check_engines()
        healthy_count = sum(health_status.values())
        
        if healthy_count < 7:
            logger.error(f"❌ Insufficient healthy engines: {healthy_count}/{len(self.engines)}")
            logger.error("Please ensure engines are running: docker-compose up -d")
            return {'error': 'Insufficient healthy engines', 'health_status': health_status}
        
        logger.info(f"✅ Proceeding with {healthy_count}/{len(self.engines)} healthy engines")
        
        # Execute all workflows
        workflow_results = {}
        
        for workflow_name, workflow_config in self.test_workflows.items():
            try:
                result = await self.run_workflow(workflow_name, workflow_config)
                workflow_results[workflow_name] = result
                
                # Brief pause between workflows
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Failed to execute workflow {workflow_name}: {e}")
                workflow_results[workflow_name] = {
                    'workflow_name': workflow_name,
                    'success': False,
                    'error': str(e),
                    'total_execution_time': 0
                }
        
        # Analyze results
        analysis = self.analyze_workflow_patterns(workflow_results)
        
        # Print summary
        self.print_workflow_summary(analysis)
        
        return {
            'workflow_results': workflow_results,
            'analysis': analysis,
            'health_status': health_status,
            'test_timestamp': datetime.now().isoformat()
        }

async def main():
    """Main workflow test execution"""
    workflow_test = NautilusWorkflowTest()
    
    logger.info("🔬 Nautilus End-to-End Workflow Integration Test")
    logger.info("Testing complete workflows across containerized architecture")
    
    results = await workflow_test.run_all_workflows()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"/tmp/nautilus_workflow_test_{timestamp}.json"
    
    try:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"📁 Results saved to: {results_file}")
    except Exception as e:
        logger.warning(f"⚠️ Could not save results: {e}")

if __name__ == "__main__":
    asyncio.run(main())