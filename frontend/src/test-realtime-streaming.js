#!/usr/bin/env node
/**
 * Real-time Data Streaming Test for MarketData Hub
 * Tests frontend's ability to receive and process live market data streams
 */

import WebSocket from 'ws';
import fetch from 'node-fetch';

class RealTimeStreamingTester {
  constructor() {
    this.baseURL = 'http://localhost:8001';
    this.wsURL = 'ws://localhost:8001';
    this.testResults = {
      streamingTests: [],
      performanceMetrics: {
        averageLatency: 0,
        maxLatency: 0,
        minLatency: Infinity,
        totalMessages: 0,
        messagesPerSecond: 0,
        connectionStability: 0,
        dataIntegrity: 0
      },
      connectedStreams: 0,
      totalStreams: 0,
      streamingCapabilities: {}
    };
  }

  async testStreamConnection(endpoint, testName, duration = 10000) {
    console.log(`🌊 Testing stream: ${testName}`);
    console.log(`   Endpoint: ${endpoint}`);
    console.log(`   Duration: ${duration/1000}s`);
    
    return new Promise((resolve) => {
      const startTime = Date.now();
      const wsUrl = `${this.wsURL}${endpoint}`;
      const ws = new WebSocket(wsUrl);
      
      let messageCount = 0;
      let latencies = [];
      let connectionLost = false;
      let firstMessageTime = null;
      let lastMessageTime = null;
      const receivedMessages = [];

      const testTimeout = setTimeout(() => {
        ws.close();
      }, duration);

      ws.on('open', () => {
        const connectionTime = Date.now() - startTime;
        console.log(`   ✅ Stream connected in ${connectionTime}ms`);
        
        // Send initial subscription message for market data streams
        if (endpoint.includes('market-data') || endpoint.includes('volatility')) {
          ws.send(JSON.stringify({
            type: 'subscribe',
            symbol: 'AAPL',
            timestamp: new Date().toISOString()
          }));
        }
      });

      ws.on('message', (data) => {
        const receiveTime = Date.now();
        messageCount++;
        
        try {
          const message = JSON.parse(data);
          
          // Calculate latency if message has timestamp
          if (message.timestamp) {
            const messageTime = new Date(message.timestamp).getTime();
            const latency = receiveTime - messageTime;
            if (latency > 0 && latency < 60000) { // Valid latency (< 1 minute)
              latencies.push(latency);
            }
          }
          
          if (!firstMessageTime) firstMessageTime = receiveTime;
          lastMessageTime = receiveTime;
          
          receivedMessages.push({
            type: message.type,
            timestamp: message.timestamp,
            receiveTime,
            dataSize: data.length
          });
          
          if (messageCount <= 3) {
            console.log(`   📨 Message ${messageCount}: ${message.type} (${data.length} bytes)`);
          } else if (messageCount === 4) {
            console.log(`   📊 Receiving messages... (${messageCount} so far)`);
          }

        } catch (error) {
          console.log(`   ⚠️ Failed to parse message: ${error.message}`);
        }
      });

      ws.on('close', () => {
        clearTimeout(testTimeout);
        
        const totalDuration = Date.now() - startTime;
        const streamDuration = lastMessageTime ? lastMessageTime - firstMessageTime : 0;
        const avgLatency = latencies.length > 0 ? latencies.reduce((a, b) => a + b, 0) / latencies.length : 0;
        const messagesPerSecond = streamDuration > 0 ? (messageCount / (streamDuration / 1000)) : 0;
        
        const result = {
          testName,
          endpoint,
          success: messageCount > 0,
          messageCount,
          totalDuration,
          streamDuration,
          averageLatency: avgLatency,
          maxLatency: latencies.length > 0 ? Math.max(...latencies) : 0,
          minLatency: latencies.length > 0 ? Math.min(...latencies) : 0,
          messagesPerSecond,
          connectionLost,
          firstMessageTime,
          lastMessageTime,
          uniqueMessageTypes: [...new Set(receivedMessages.map(m => m.type))],
          totalDataReceived: receivedMessages.reduce((sum, m) => sum + m.dataSize, 0)
        };

        if (result.success) {
          console.log(`   ✅ Stream completed: ${messageCount} messages, ${messagesPerSecond.toFixed(1)} msg/s`);
          this.testResults.connectedStreams++;
        } else {
          console.log(`   ❌ No messages received`);
        }

        this.testResults.totalStreams++;
        this.testResults.streamingTests.push(result);
        resolve(result);
      });

      ws.on('error', (error) => {
        clearTimeout(testTimeout);
        connectionLost = true;
        console.log(`   💥 Stream error: ${error.message}`);
        
        resolve({
          testName,
          endpoint,
          success: false,
          error: error.message,
          messageCount,
          totalDuration: Date.now() - startTime,
          connectionLost: true
        });
      });
    });
  }

  async testDataPersistence() {
    console.log('\n💾 Testing Data Persistence and Consistency');
    
    // Test if backend maintains data consistency across requests
    const symbols = ['AAPL', 'GOOGL', 'MSFT'];
    const consistencyResults = [];

    for (const symbol of symbols) {
      try {
        // Make multiple requests for the same data
        const requests = await Promise.all([
          fetch(`${this.baseURL}/api/v1/nautilus-data/alpha-vantage/search?keywords=${symbol}`),
          fetch(`${this.baseURL}/api/v1/nautilus-data/alpha-vantage/search?keywords=${symbol}`),
          fetch(`${this.baseURL}/api/v1/nautilus-data/alpha-vantage/search?keywords=${symbol}`)
        ]);

        const responses = await Promise.all(requests.map(r => r.json()));
        
        // Check consistency
        const isConsistent = responses.every(response => 
          JSON.stringify(response) === JSON.stringify(responses[0])
        );

        consistencyResults.push({
          symbol,
          consistent: isConsistent,
          responseCount: responses.length,
          responseSize: JSON.stringify(responses[0]).length
        });

        console.log(`   ${isConsistent ? '✅' : '❌'} ${symbol}: Data consistency check`);

      } catch (error) {
        console.log(`   💥 ${symbol}: Error testing consistency - ${error.message}`);
        consistencyResults.push({
          symbol,
          consistent: false,
          error: error.message
        });
      }
    }

    this.testResults.dataConsistency = consistencyResults;
    return consistencyResults;
  }

  async runStreamingTestSuite() {
    console.log('🌊 Starting Real-time Data Streaming Test Suite');
    console.log('=' .repeat(60));

    // Phase 1: Test working WebSocket endpoints
    console.log('\n📡 Phase 1: WebSocket Stream Tests');
    
    const workingEndpoints = [
      { endpoint: '/ws/system/health', name: 'System Health Stream' },
      { endpoint: '/ws/messagebus', name: 'MessageBus Event Stream' }
    ];

    for (const { endpoint, name } of workingEndpoints) {
      await this.testStreamConnection(endpoint, name, 8000);
    }

    // Phase 2: Test market data endpoints (these may not work but we test them)
    console.log('\n📈 Phase 2: Market Data Stream Tests');
    
    const marketDataEndpoints = [
      { endpoint: '/api/v1/ws/market-data/AAPL', name: 'Market Data Stream (AAPL)' },
      { endpoint: '/api/v1/volatility/ws/streaming/AAPL', name: 'Volatility Stream (AAPL)' }
    ];

    for (const { endpoint, name } of marketDataEndpoints) {
      await this.testStreamConnection(endpoint, name, 5000);
    }

    // Phase 3: Data Persistence and Consistency
    await this.testDataPersistence();

    // Phase 4: Frontend Integration Simulation
    console.log('\n🖥️ Phase 4: Frontend Integration Simulation');
    await this.simulateFrontendIntegration();

    // Calculate final metrics
    this.calculateMetrics();
    this.printResults();

    return this.testResults;
  }

  async simulateFrontendIntegration() {
    console.log('   Testing frontend data consumption patterns...');
    
    // Simulate typical frontend requests that would happen during streaming
    const frontendPatterns = [
      { name: 'Initial Dashboard Load', requests: ['/health', '/api/v1/nautilus-data/health'] },
      { name: 'Market Data Refresh', requests: ['/api/v1/nautilus-data/alpha-vantage/search?keywords=AAPL'] },
      { name: 'Portfolio Update', requests: ['/api/v1/portfolio/balance', '/api/v1/exchanges/status'] }
    ];

    const integrationResults = [];

    for (const pattern of frontendPatterns) {
      const startTime = Date.now();
      
      try {
        const requests = pattern.requests.map(endpoint => 
          fetch(`${this.baseURL}${endpoint}`, { timeout: 5000 })
        );
        
        const responses = await Promise.all(requests);
        const duration = Date.now() - startTime;
        const allSuccessful = responses.every(r => r.ok);

        integrationResults.push({
          pattern: pattern.name,
          success: allSuccessful,
          duration,
          requests: pattern.requests.length,
          successRate: responses.filter(r => r.ok).length / responses.length
        });

        console.log(`   ${allSuccessful ? '✅' : '❌'} ${pattern.name}: ${duration}ms (${pattern.requests.length} requests)`);

      } catch (error) {
        console.log(`   💥 ${pattern.name}: ${error.message}`);
        integrationResults.push({
          pattern: pattern.name,
          success: false,
          error: error.message
        });
      }
    }

    this.testResults.frontendIntegration = integrationResults;
  }

  calculateMetrics() {
    const successfulStreams = this.testResults.streamingTests.filter(s => s.success);
    
    if (successfulStreams.length > 0) {
      this.testResults.performanceMetrics.averageLatency = 
        successfulStreams.reduce((sum, s) => sum + (s.averageLatency || 0), 0) / successfulStreams.length;
      
      this.testResults.performanceMetrics.maxLatency = 
        Math.max(...successfulStreams.map(s => s.maxLatency || 0));
      
      this.testResults.performanceMetrics.minLatency = 
        Math.min(...successfulStreams.map(s => s.minLatency || Infinity));
      
      this.testResults.performanceMetrics.totalMessages = 
        successfulStreams.reduce((sum, s) => sum + s.messageCount, 0);
      
      this.testResults.performanceMetrics.messagesPerSecond = 
        successfulStreams.reduce((sum, s) => sum + (s.messagesPerSecond || 0), 0) / successfulStreams.length;
    }

    this.testResults.performanceMetrics.connectionStability = 
      (this.testResults.connectedStreams / this.testResults.totalStreams) * 100;
  }

  printResults() {
    console.log('\n📊 Real-time Streaming Test Results');
    console.log('=' .repeat(60));
    
    const connected = this.testResults.connectedStreams;
    const total = this.testResults.totalStreams;
    const connectionRate = (connected / total) * 100;
    
    console.log(`🌊 Stream Connections: ${connected}/${total} successful (${connectionRate.toFixed(1)}%)`);
    
    if (connected > 0) {
      console.log(`⚡ Streaming Performance:`);
      console.log(`   Average Latency: ${this.testResults.performanceMetrics.averageLatency.toFixed(2)}ms`);
      console.log(`   Messages/Second: ${this.testResults.performanceMetrics.messagesPerSecond.toFixed(1)}`);
      console.log(`   Total Messages: ${this.testResults.performanceMetrics.totalMessages}`);
    }

    // Show stream results
    console.log('\n📋 Stream Test Details:');
    this.testResults.streamingTests.forEach((result, index) => {
      const status = result.success ? '✅' : '❌';
      const msgInfo = result.success ? `, ${result.messageCount} msgs` : '';
      console.log(`   ${index + 1}. ${status} ${result.testName}${msgInfo}`);
      if (!result.success && result.error) {
        console.log(`      Error: ${result.error}`);
      }
    });

    // Data consistency results
    if (this.testResults.dataConsistency) {
      const consistent = this.testResults.dataConsistency.filter(r => r.consistent).length;
      const totalChecks = this.testResults.dataConsistency.length;
      console.log(`\n💾 Data Consistency: ${consistent}/${totalChecks} symbols consistent`);
    }

    // Frontend integration results
    if (this.testResults.frontendIntegration) {
      const successful = this.testResults.frontendIntegration.filter(r => r.success).length;
      const totalPatterns = this.testResults.frontendIntegration.length;
      console.log(`🖥️ Frontend Integration: ${successful}/${totalPatterns} patterns working`);
    }

    // Real-time capability assessment
    console.log('\n🎯 Real-time Streaming Assessment:');
    if (connectionRate >= 75 && this.testResults.performanceMetrics.totalMessages > 20) {
      console.log('   ✅ EXCELLENT: Real-time streaming capabilities are robust');
    } else if (connectionRate >= 50) {
      console.log('   ✅ GOOD: Basic streaming functionality is working');
    } else if (connectionRate >= 25) {
      console.log('   ⚠️ PARTIAL: Limited streaming capabilities');
    } else {
      console.log('   ❌ POOR: Streaming functionality needs significant work');
    }

    if (this.testResults.performanceMetrics.averageLatency < 100) {
      console.log('   ⚡ LATENCY: Excellent real-time performance');
    } else if (this.testResults.performanceMetrics.averageLatency < 500) {
      console.log('   ⚡ LATENCY: Good real-time performance');
    } else {
      console.log('   ⚠️ LATENCY: Real-time performance could be improved');
    }
  }
}

// Main execution
async function main() {
  const tester = new RealTimeStreamingTester();
  
  try {
    await tester.runStreamingTestSuite();
    
    const connectionRate = (tester.testResults.connectedStreams / tester.testResults.totalStreams) * 100;
    
    if (connectionRate >= 50) {
      console.log('\n🎉 Real-time streaming tests completed successfully!');
      process.exit(0);
    } else {
      console.log('\n⚠️ Real-time streaming tests completed with issues.');
      process.exit(1);
    }
    
  } catch (error) {
    console.error('\n💥 Streaming test suite failed:', error.message);
    process.exit(1);
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}

export default RealTimeStreamingTester;