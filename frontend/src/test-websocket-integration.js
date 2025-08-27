#!/usr/bin/env node
/**
 * WebSocket Integration Testing for MarketData Hub
 * Tests real-time streaming performance and connectivity
 */

// Use built-in WebSocket API for Node.js 20+
import WebSocket from 'ws';

class WebSocketTester {
  constructor() {
    this.connections = new Map();
    this.testResults = {
      connectionTests: [],
      performanceMetrics: {
        averageLatency: 0,
        maxLatency: 0,
        minLatency: Infinity,
        totalMessages: 0,
        failedConnections: 0,
        successfulConnections: 0
      }
    };
  }

  async testConnection(endpoint, testName) {
    console.log(`\n🔗 Testing WebSocket connection: ${testName}`);
    console.log(`   Endpoint: ${endpoint}`);
    
    return new Promise((resolve) => {
      const startTime = Date.now();
      const ws = new WebSocket(endpoint);
      let isResolved = false;

      const timeout = setTimeout(() => {
        if (!isResolved) {
          isResolved = true;
          this.testResults.performanceMetrics.failedConnections++;
          resolve({
            testName,
            endpoint,
            success: false,
            error: 'Connection timeout',
            duration: Date.now() - startTime
          });
        }
      }, 10000);

      ws.on('open', () => {
        if (!isResolved) {
          const duration = Date.now() - startTime;
          console.log(`   ✅ Connected in ${duration}ms`);
          
          clearTimeout(timeout);
          isResolved = true;
          this.testResults.performanceMetrics.successfulConnections++;
          
          resolve({
            testName,
            endpoint,
            success: true,
            duration,
            connectionTime: duration
          });
          
          ws.close();
        }
      });

      ws.on('error', (error) => {
        if (!isResolved) {
          console.log(`   ❌ Connection failed: ${error.message}`);
          
          clearTimeout(timeout);
          isResolved = true;
          this.testResults.performanceMetrics.failedConnections++;
          
          resolve({
            testName,
            endpoint,
            success: false,
            error: error.message,
            duration: Date.now() - startTime
          });
        }
      });

      ws.on('close', () => {
        clearTimeout(timeout);
      });
    });
  }

  async testMessageLatency(endpoint, testName, messageCount = 10) {
    console.log(`\n⚡ Testing message latency: ${testName}`);
    console.log(`   Sending ${messageCount} test messages`);
    
    return new Promise((resolve) => {
      const ws = new WebSocket(endpoint);
      let latencies = [];
      let messagesSent = 0;
      let messagesReceived = 0;
      const messageTimes = new Map();

      ws.on('open', () => {
        console.log(`   ✅ Connection established, sending test messages...`);
        
        for (let i = 0; i < messageCount; i++) {
          const messageId = `test_msg_${i}_${Date.now()}`;
          const sendTime = Date.now();
          messageTimes.set(messageId, sendTime);
          
          ws.send(JSON.stringify({
            type: 'ping',
            messageId,
            timestamp: new Date().toISOString()
          }));
          
          messagesSent++;
        }
      });

      ws.on('message', (data) => {
        try {
          const message = JSON.parse(data);
          const receiveTime = Date.now();
          
          if (message.messageId && messageTimes.has(message.messageId)) {
            const sendTime = messageTimes.get(message.messageId);
            const latency = receiveTime - sendTime;
            latencies.push(latency);
            messagesReceived++;
            
            console.log(`   📨 Message ${messagesReceived}/${messageCount}: ${latency}ms latency`);
          }
          
          // Complete test when all messages received
          if (messagesReceived >= messageCount || messagesReceived >= messagesSent) {
            ws.close();
            
            const avgLatency = latencies.reduce((a, b) => a + b, 0) / latencies.length;
            const maxLatency = Math.max(...latencies);
            const minLatency = Math.min(...latencies);
            
            resolve({
              testName,
              endpoint,
              success: true,
              messagesSent,
              messagesReceived,
              averageLatency: avgLatency,
              maxLatency,
              minLatency,
              latencies
            });
          }
        } catch (error) {
          console.log(`   ⚠️ Failed to parse message: ${error.message}`);
        }
      });

      ws.on('error', (error) => {
        resolve({
          testName,
          endpoint,
          success: false,
          error: error.message,
          messagesSent,
          messagesReceived
        });
      });

      // Timeout after 30 seconds
      setTimeout(() => {
        ws.close();
        resolve({
          testName,
          endpoint,
          success: false,
          error: 'Test timeout',
          messagesSent,
          messagesReceived
        });
      }, 30000);
    });
  }

  async runFullTestSuite() {
    console.log('🚀 Starting WebSocket Integration Test Suite');
    console.log('=' .repeat(60));

    // Test endpoints based on the WebSocket client configuration
    const testEndpoints = [
      {
        name: 'System Health WebSocket',
        url: 'ws://localhost:8001/ws/system/health'
      },
      {
        name: 'Trade Updates WebSocket',
        url: 'ws://localhost:8001/ws/trades/updates'
      },
      {
        name: 'MessageBus WebSocket',
        url: 'ws://localhost:8001/ws/messagebus'
      },
      {
        name: 'Market Data WebSocket (AAPL)',
        url: 'ws://localhost:8001/api/v1/ws/market-data/AAPL'
      },
      {
        name: 'Volatility WebSocket (AAPL)',
        url: 'ws://localhost:8001/api/v1/volatility/ws/streaming/AAPL'
      }
    ];

    // Test basic connectivity
    console.log('\n📡 Phase 1: Basic Connectivity Tests');
    for (const endpoint of testEndpoints) {
      const result = await this.testConnection(endpoint.url, endpoint.name);
      this.testResults.connectionTests.push(result);
    }

    // Test message latency for successful connections
    console.log('\n⚡ Phase 2: Message Latency Tests');
    const successfulEndpoints = this.testResults.connectionTests.filter(r => r.success);
    
    if (successfulEndpoints.length > 0) {
      for (const endpoint of successfulEndpoints.slice(0, 3)) { // Test top 3 working endpoints
        const latencyResult = await this.testMessageLatency(endpoint.endpoint, endpoint.testName, 5);
        
        if (latencyResult.success) {
          this.testResults.performanceMetrics.totalMessages += latencyResult.messagesReceived;
          this.testResults.performanceMetrics.averageLatency = 
            (this.testResults.performanceMetrics.averageLatency + latencyResult.averageLatency) / 2;
          this.testResults.performanceMetrics.maxLatency = 
            Math.max(this.testResults.performanceMetrics.maxLatency, latencyResult.maxLatency);
          this.testResults.performanceMetrics.minLatency = 
            Math.min(this.testResults.performanceMetrics.minLatency, latencyResult.minLatency);
        }
      }
    } else {
      console.log('   ⚠️ No successful connections available for latency testing');
    }

    // Print comprehensive results
    this.printResults();
    
    return this.testResults;
  }

  printResults() {
    console.log('\n📊 WebSocket Integration Test Results');
    console.log('=' .repeat(60));
    
    const successful = this.testResults.connectionTests.filter(r => r.success).length;
    const total = this.testResults.connectionTests.length;
    const successRate = ((successful / total) * 100).toFixed(1);
    
    console.log(`🔗 Connection Tests: ${successful}/${total} successful (${successRate}%)`);
    
    if (successful > 0) {
      console.log(`⚡ Performance Metrics:`);
      console.log(`   Average Latency: ${this.testResults.performanceMetrics.averageLatency.toFixed(2)}ms`);
      console.log(`   Min Latency: ${this.testResults.performanceMetrics.minLatency.toFixed(2)}ms`);
      console.log(`   Max Latency: ${this.testResults.performanceMetrics.maxLatency.toFixed(2)}ms`);
      console.log(`   Total Messages: ${this.testResults.performanceMetrics.totalMessages}`);
    }
    
    console.log('\n📋 Detailed Results:');
    this.testResults.connectionTests.forEach((result, index) => {
      const status = result.success ? '✅' : '❌';
      const duration = result.duration || result.connectionTime || 0;
      console.log(`   ${index + 1}. ${status} ${result.testName}: ${duration}ms`);
      if (!result.success && result.error) {
        console.log(`      Error: ${result.error}`);
      }
    });
    
    // Integration Assessment
    console.log('\n🎯 Integration Assessment:');
    if (successRate >= 80) {
      console.log('   ✅ EXCELLENT: WebSocket integration is working well');
    } else if (successRate >= 60) {
      console.log('   ⚠️ GOOD: Most WebSocket endpoints are functional');
    } else if (successRate >= 40) {
      console.log('   ⚠️ PARTIAL: Some WebSocket connectivity issues detected');
    } else {
      console.log('   ❌ POOR: Significant WebSocket connectivity problems');
    }
  }
}

// Run the test suite
async function main() {
  const tester = new WebSocketTester();
  
  try {
    await tester.runFullTestSuite();
    
    // Calculate exit code based on success rate
    const successful = tester.testResults.connectionTests.filter(r => r.success).length;
    const total = tester.testResults.connectionTests.length;
    const successRate = (successful / total) * 100;
    
    if (successRate >= 60) {
      console.log('\n🎉 WebSocket integration tests completed successfully!');
      process.exit(0);
    } else {
      console.log('\n⚠️ WebSocket integration tests completed with issues.');
      process.exit(1);
    }
    
  } catch (error) {
    console.error('\n💥 Test suite failed:', error.message);
    process.exit(1);
  }
}

// Run if this is the main module
if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}

export default WebSocketTester;