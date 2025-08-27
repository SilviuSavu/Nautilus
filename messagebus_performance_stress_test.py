#!/usr/bin/env python3
"""
📡 MESSAGEBUS PERFORMANCE STRESS TESTING
Dream Team Mission: Test MessageBus performance with concurrent real market data flows
"""

import asyncio
import json
import time
import redis
import threading
from datetime import datetime
from typing import Dict, List, Any
import logging
import numpy as np
import concurrent.futures
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MessageBusConnection:
    name: str
    host: str
    port: int
    db: int = 0

class MessageBusStressTester:
    """
    📡 MESSAGEBUS PERFORMANCE STRESS TESTER
    
    Features:
    - Concurrent message publishing from all engines
    - High-frequency market data simulation
    - Cross-engine communication testing
    - Latency and throughput measurement
    - Memory usage monitoring
    """
    
    def __init__(self):
        # MessageBus connections (Redis instances)
        self.connections = [
            MessageBusConnection("Main MessageBus", "localhost", 6379, 0),
            MessageBusConnection("MarketData Bus", "localhost", 6380, 0)
        ]
        
        # Test channels for different engine communications
        self.test_channels = [
            "engine_coordination",
            "market_data_feed",
            "risk_alerts", 
            "trading_signals",
            "portfolio_updates",
            "analytics_results",
            "ml_predictions",
            "strategy_decisions",
            "factor_calculations",
            "websocket_broadcasts",
            "collateral_monitoring",
            "vpin_alerts",
            "backtesting_results"
        ]
        
        self.results = []
        logger.info("📡 MessageBus Stress Tester initialized")
        logger.info(f"   Connections to test: {len(self.connections)}")
        logger.info(f"   Channels to test: {len(self.test_channels)}")

    def connect_to_messagebus(self, connection: MessageBusConnection) -> redis.Redis:
        """Connect to a MessageBus (Redis) instance"""
        try:
            client = redis.Redis(
                host=connection.host,
                port=connection.port,
                db=connection.db,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            client.ping()
            return client
        except Exception as e:
            logger.error(f"Failed to connect to {connection.name}: {e}")
            return None

    def generate_market_data_message(self, symbol: str, message_id: int) -> Dict[str, Any]:
        """Generate realistic market data message"""
        return {
            "message_id": message_id,
            "timestamp": time.time(),
            "symbol": symbol,
            "price": round(np.random.uniform(100, 500), 2),
            "volume": np.random.randint(100, 10000),
            "bid": round(np.random.uniform(99, 499), 2),
            "ask": round(np.random.uniform(101, 501), 2),
            "volatility": round(np.random.uniform(0.1, 0.5), 3),
            "source": "stress_test",
            "engine": "market_data_simulator"
        }

    def generate_engine_message(self, engine_name: str, message_type: str, message_id: int) -> Dict[str, Any]:
        """Generate engine coordination message"""
        return {
            "message_id": message_id,
            "timestamp": time.time(),
            "engine": engine_name,
            "message_type": message_type,
            "data": {
                "status": "operational",
                "load": np.random.uniform(0.1, 0.9),
                "processed_count": np.random.randint(0, 1000),
                "response_time_ms": round(np.random.uniform(1, 50), 2),
                "memory_usage_mb": np.random.randint(100, 1000)
            },
            "priority": np.random.choice(["low", "normal", "high"])
        }

    def publish_messages_to_channel(self, client: redis.Redis, channel: str, num_messages: int, message_type: str) -> Dict[str, Any]:
        """Publish messages to a specific channel"""
        start_time = time.time()
        successful_publishes = 0
        failed_publishes = 0
        total_latency = 0
        
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "SPY", "QQQ", "IWM", "VIX"]
        engines = ["analytics", "risk", "factor", "ml", "portfolio", "strategy", "websocket", "collateral", "vpin", "backtesting"]
        
        for i in range(num_messages):
            try:
                # Generate message based on type
                if message_type == "market_data":
                    symbol = symbols[i % len(symbols)]
                    message = self.generate_market_data_message(symbol, i)
                else:
                    engine = engines[i % len(engines)]
                    message = self.generate_engine_message(engine, message_type, i)
                
                # Measure publish latency
                publish_start = time.time()
                result = client.publish(channel, json.dumps(message))
                publish_end = time.time()
                
                publish_latency = (publish_end - publish_start) * 1000  # Convert to ms
                total_latency += publish_latency
                
                if result >= 0:  # Redis returns number of subscribers
                    successful_publishes += 1
                else:
                    failed_publishes += 1
                    
            except Exception as e:
                failed_publishes += 1
                logger.error(f"Publish error on {channel}: {e}")
        
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            "channel": channel,
            "message_type": message_type,
            "total_messages": num_messages,
            "successful_publishes": successful_publishes,
            "failed_publishes": failed_publishes,
            "duration_seconds": duration,
            "messages_per_second": successful_publishes / duration if duration > 0 else 0,
            "average_latency_ms": total_latency / successful_publishes if successful_publishes > 0 else 0,
            "success_rate_percent": (successful_publishes / num_messages) * 100
        }

    def test_concurrent_publishing(self, client: redis.Redis, num_channels: int = 13, messages_per_channel: int = 1000) -> Dict[str, Any]:
        """Test concurrent publishing across multiple channels"""
        logger.info(f"📡 Testing concurrent publishing: {num_channels} channels, {messages_per_channel} messages each")
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_channels) as executor:
            futures = []
            
            for i in range(num_channels):
                channel = self.test_channels[i % len(self.test_channels)]
                message_type = "market_data" if "market" in channel else "engine_coordination"
                
                future = executor.submit(
                    self.publish_messages_to_channel,
                    client,
                    channel,
                    messages_per_channel,
                    message_type
                )
                futures.append(future)
            
            # Collect results
            channel_results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    channel_results.append(result)
                except Exception as e:
                    logger.error(f"Channel publishing error: {e}")
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Aggregate statistics
        total_messages = sum(r["total_messages"] for r in channel_results)
        total_successful = sum(r["successful_publishes"] for r in channel_results)
        total_failed = sum(r["failed_publishes"] for r in channel_results)
        
        average_latency = np.mean([r["average_latency_ms"] for r in channel_results if r["average_latency_ms"] > 0])
        peak_throughput = max([r["messages_per_second"] for r in channel_results])
        total_throughput = total_successful / total_duration if total_duration > 0 else 0
        
        return {
            "test_name": "concurrent_publishing",
            "channels_tested": num_channels,
            "messages_per_channel": messages_per_channel,
            "total_messages": total_messages,
            "successful_messages": total_successful,
            "failed_messages": total_failed,
            "success_rate_percent": (total_successful / total_messages) * 100,
            "total_duration_seconds": total_duration,
            "total_throughput_msgs_per_sec": total_throughput,
            "peak_channel_throughput_msgs_per_sec": peak_throughput,
            "average_latency_ms": average_latency,
            "channel_results": channel_results
        }

    def test_subscriber_performance(self, client: redis.Redis, test_duration: int = 30) -> Dict[str, Any]:
        """Test subscriber performance and message delivery"""
        logger.info(f"📥 Testing subscriber performance for {test_duration} seconds")
        
        received_messages = []
        subscriber_stats = {"connected": False, "messages_received": 0, "errors": 0}
        
        def subscriber_worker():
            try:
                pubsub = client.pubsub()
                pubsub.subscribe(*self.test_channels[:5])  # Subscribe to first 5 channels
                subscriber_stats["connected"] = True
                
                start_time = time.time()
                while time.time() - start_time < test_duration:
                    message = pubsub.get_message(timeout=1)
                    if message and message['type'] == 'message':
                        received_messages.append({
                            "timestamp": time.time(),
                            "channel": message['channel'],
                            "data_size": len(message['data'])
                        })
                        subscriber_stats["messages_received"] += 1
                
                pubsub.close()
            except Exception as e:
                subscriber_stats["errors"] += 1
                logger.error(f"Subscriber error: {e}")
        
        # Start subscriber in background
        subscriber_thread = threading.Thread(target=subscriber_worker)
        subscriber_thread.start()
        
        # Wait a moment for subscriber to connect
        time.sleep(1)
        
        # Publish test messages
        test_messages = 1000
        publish_results = self.test_concurrent_publishing(client, 5, test_messages // 5)
        
        # Wait for subscriber to finish
        subscriber_thread.join(timeout=test_duration + 5)
        
        return {
            "test_name": "subscriber_performance",
            "test_duration_seconds": test_duration,
            "subscriber_connected": subscriber_stats["connected"],
            "messages_published": publish_results["successful_messages"],
            "messages_received": subscriber_stats["messages_received"],
            "message_delivery_rate_percent": (subscriber_stats["messages_received"] / publish_results["successful_messages"]) * 100 if publish_results["successful_messages"] > 0 else 0,
            "subscriber_errors": subscriber_stats["errors"],
            "publishing_results": publish_results
        }

    def test_high_frequency_scenario(self, client: redis.Redis) -> Dict[str, Any]:
        """Test high-frequency trading scenario simulation"""
        logger.info("⚡ Testing high-frequency trading scenario")
        
        # Simulate flash crash scenario with high message volume
        symbols = ["SPY", "QQQ", "IWM", "VIX"]
        channels = ["market_data_feed", "risk_alerts", "trading_signals"]
        
        start_time = time.time()
        results = []
        
        # High-frequency burst: 10,000 messages in 60 seconds
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(channels)) as executor:
            futures = []
            
            for channel in channels:
                future = executor.submit(
                    self.publish_messages_to_channel,
                    client,
                    channel,
                    3333,  # ~3333 messages per channel = ~10k total
                    "market_data"
                )
                futures.append(future)
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"HFT scenario error: {e}")
        
        end_time = time.time()
        
        total_messages = sum(r["total_messages"] for r in results)
        total_successful = sum(r["successful_publishes"] for r in results)
        total_throughput = total_successful / (end_time - start_time)
        
        return {
            "test_name": "high_frequency_trading_scenario",
            "total_messages": total_messages,
            "successful_messages": total_successful,
            "duration_seconds": end_time - start_time,
            "throughput_msgs_per_second": total_throughput,
            "target_achieved": total_throughput > 100,  # Target: >100 msgs/sec
            "channel_results": results
        }

    def test_messagebus_memory_usage(self, client: redis.Redis) -> Dict[str, Any]:
        """Test MessageBus memory usage under load"""
        logger.info("💾 Testing MessageBus memory usage under load")
        
        try:
            # Get initial memory usage
            initial_info = client.info('memory')
            initial_memory = initial_info.get('used_memory', 0)
            
            # Execute intensive messaging
            load_test_results = self.test_concurrent_publishing(client, 13, 5000)
            
            # Get final memory usage
            final_info = client.info('memory')
            final_memory = final_info.get('used_memory', 0)
            
            memory_increase = final_memory - initial_memory
            memory_increase_percent = (memory_increase / initial_memory) * 100 if initial_memory > 0 else 0
            
            return {
                "test_name": "messagebus_memory_usage",
                "initial_memory_bytes": initial_memory,
                "final_memory_bytes": final_memory,
                "memory_increase_bytes": memory_increase,
                "memory_increase_percent": memory_increase_percent,
                "load_test_results": load_test_results,
                "memory_efficiency": "good" if memory_increase_percent < 50 else "concerning"
            }
        except Exception as e:
            return {
                "test_name": "messagebus_memory_usage",
                "error": str(e),
                "status": "failed"
            }

    async def run_comprehensive_messagebus_stress_test(self) -> Dict[str, Any]:
        """Execute comprehensive MessageBus stress testing"""
        logger.info("📡 STARTING COMPREHENSIVE MESSAGEBUS STRESS TEST")
        logger.info("=" * 70)
        
        mission_start = time.time()
        all_results = {}
        
        # Test each MessageBus connection
        for connection in self.connections:
            logger.info(f"\n🔌 Testing {connection.name} ({connection.host}:{connection.port})")
            
            client = self.connect_to_messagebus(connection)
            if not client:
                all_results[connection.name] = {"status": "connection_failed"}
                continue
            
            connection_results = {}
            
            try:
                # Phase 1: Concurrent Publishing
                logger.info("  📤 Phase 1: Concurrent Publishing Test")
                concurrent_results = self.test_concurrent_publishing(client, 13, 1000)
                connection_results["concurrent_publishing"] = concurrent_results
                
                # Phase 2: Subscriber Performance
                logger.info("  📥 Phase 2: Subscriber Performance Test")
                subscriber_results = self.test_subscriber_performance(client, 30)
                connection_results["subscriber_performance"] = subscriber_results
                
                # Phase 3: High-Frequency Scenario
                logger.info("  ⚡ Phase 3: High-Frequency Trading Scenario")
                hft_results = self.test_high_frequency_scenario(client)
                connection_results["high_frequency_scenario"] = hft_results
                
                # Phase 4: Memory Usage
                logger.info("  💾 Phase 4: Memory Usage Test")
                memory_results = self.test_messagebus_memory_usage(client)
                connection_results["memory_usage"] = memory_results
                
                all_results[connection.name] = connection_results
                
            except Exception as e:
                logger.error(f"Error testing {connection.name}: {e}")
                all_results[connection.name] = {"error": str(e), "status": "failed"}
            finally:
                client.close()
        
        mission_end = time.time()
        
        # Compile final results
        final_results = {
            "mission": "comprehensive_messagebus_stress_test",
            "execution_time_seconds": mission_end - mission_start,
            "timestamp": datetime.now().isoformat(),
            "connections_tested": len(self.connections),
            "channels_tested": len(self.test_channels),
            "results_by_connection": all_results,
            "overall_performance": self._calculate_overall_performance(all_results)
        }
        
        # Log summary
        logger.info("\n" + "=" * 70)
        logger.info("📡 MESSAGEBUS STRESS TEST COMPLETE")
        logger.info("=" * 70)
        logger.info(f"⏱️  Total Duration: {final_results['execution_time_seconds']:.1f} seconds")
        logger.info(f"🔌 Connections Tested: {len(self.connections)}")
        logger.info(f"📡 Channels Tested: {len(self.test_channels)}")
        
        # Log performance summary for each connection
        for conn_name, results in all_results.items():
            if "concurrent_publishing" in results:
                pub_results = results["concurrent_publishing"]
                logger.info(f"  {conn_name}:")
                logger.info(f"    📤 Throughput: {pub_results['total_throughput_msgs_per_sec']:.1f} msgs/sec")
                logger.info(f"    ⏱️  Latency: {pub_results.get('average_latency_ms', 0):.2f}ms avg")
                logger.info(f"    ✅ Success Rate: {pub_results['success_rate_percent']:.1f}%")
        
        return final_results

    def _calculate_overall_performance(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall performance metrics"""
        total_throughput = 0
        avg_latencies = []
        success_rates = []
        
        for conn_name, results in all_results.items():
            if "concurrent_publishing" in results:
                pub_results = results["concurrent_publishing"]
                total_throughput += pub_results.get("total_throughput_msgs_per_sec", 0)
                
                if pub_results.get("average_latency_ms", 0) > 0:
                    avg_latencies.append(pub_results["average_latency_ms"])
                
                success_rates.append(pub_results.get("success_rate_percent", 0))
        
        return {
            "total_system_throughput_msgs_per_sec": total_throughput,
            "average_system_latency_ms": np.mean(avg_latencies) if avg_latencies else 0,
            "average_success_rate_percent": np.mean(success_rates) if success_rates else 0,
            "connections_operational": len([r for r in all_results.values() if "error" not in r]),
            "performance_grade": "excellent" if total_throughput > 1000 and np.mean(avg_latencies) < 10 else "good" if total_throughput > 500 else "needs_improvement"
        }

    def save_results(self, results: Dict[str, Any]) -> str:
        """Save test results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"messagebus_stress_test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"📁 Results saved to: {filename}")
        return filename

async def main():
    """Main execution"""
    tester = MessageBusStressTester()
    
    # Execute comprehensive stress test
    results = await tester.run_comprehensive_messagebus_stress_test()
    
    # Save results
    filename = tester.save_results(results)
    
    print(f"\n📡 MESSAGEBUS STRESS TEST COMPLETE!")
    print(f"📊 Results saved to: {filename}")
    
    return results

if __name__ == "__main__":
    asyncio.run(main())