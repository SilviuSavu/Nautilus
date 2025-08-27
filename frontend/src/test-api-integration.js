#!/usr/bin/env node
/**
 * Comprehensive API Integration Testing for Nautilus Trading Platform
 * Tests frontend-backend integration across all MarketData Hub endpoints
 */

import fetch from 'node-fetch';

class APIIntegrationTester {
  constructor() {
    this.baseURL = 'http://localhost:8001';
    this.testResults = {
      endpointTests: [],
      performanceMetrics: {
        averageResponseTime: 0,
        maxResponseTime: 0,
        minResponseTime: Infinity,
        totalRequests: 0,
        successfulRequests: 0,
        failedRequests: 0,
        timeouts: 0
      },
      dataSourceStatus: {},
      systemHealth: null
    };
    this.timeout = 15000; // 15 seconds
  }

  async testEndpoint(endpoint, method = 'GET', data = null, expectedStatusCodes = [200]) {
    const startTime = Date.now();
    const testName = `${method} ${endpoint}`;
    
    console.log(`🔍 Testing: ${testName}`);
    
    try {
      const options = {
        method,
        headers: { 'Content-Type': 'application/json' },
        timeout: this.timeout
      };
      
      if (data) {
        options.body = JSON.stringify(data);
      }

      const response = await fetch(`${this.baseURL}${endpoint}`, options);
      const responseTime = Date.now() - startTime;
      const responseData = await response.text();
      
      let parsedData = null;
      try {
        parsedData = JSON.parse(responseData);
      } catch (e) {
        parsedData = responseData;
      }

      const success = expectedStatusCodes.includes(response.status);
      const result = {
        testName,
        endpoint,
        method,
        success,
        status: response.status,
        responseTime,
        dataSize: responseData.length,
        hasData: Boolean(parsedData && (Array.isArray(parsedData) ? parsedData.length > 0 : Object.keys(parsedData).length > 0)),
        error: success ? null : `HTTP ${response.status}: ${response.statusText}`
      };

      if (success) {
        console.log(`   ✅ ${response.status} - ${responseTime}ms - ${responseData.length} bytes`);
        this.testResults.performanceMetrics.successfulRequests++;
      } else {
        console.log(`   ❌ ${response.status} - ${response.statusText}`);
        this.testResults.performanceMetrics.failedRequests++;
      }

      this.updatePerformanceMetrics(responseTime);
      this.testResults.endpointTests.push(result);
      
      return { ...result, data: parsedData };

    } catch (error) {
      const responseTime = Date.now() - startTime;
      const isTimeout = error.message.includes('timeout') || responseTime >= this.timeout;
      
      if (isTimeout) {
        this.testResults.performanceMetrics.timeouts++;
        console.log(`   ⏱️ TIMEOUT - ${responseTime}ms`);
      } else {
        console.log(`   💥 ERROR - ${error.message}`);
      }
      
      const result = {
        testName,
        endpoint,
        method,
        success: false,
        status: 0,
        responseTime,
        dataSize: 0,
        hasData: false,
        error: isTimeout ? 'Request timeout' : error.message
      };

      this.testResults.performanceMetrics.failedRequests++;
      this.updatePerformanceMetrics(responseTime);
      this.testResults.endpointTests.push(result);
      
      return result;
    }
  }

  updatePerformanceMetrics(responseTime) {
    this.testResults.performanceMetrics.totalRequests++;
    this.testResults.performanceMetrics.maxResponseTime = Math.max(
      this.testResults.performanceMetrics.maxResponseTime, 
      responseTime
    );
    this.testResults.performanceMetrics.minResponseTime = Math.min(
      this.testResults.performanceMetrics.minResponseTime, 
      responseTime
    );
    
    // Running average
    const total = this.testResults.performanceMetrics.totalRequests;
    const current = this.testResults.performanceMetrics.averageResponseTime;
    this.testResults.performanceMetrics.averageResponseTime = 
      ((current * (total - 1)) + responseTime) / total;
  }

  async runFullTestSuite() {
    console.log('🚀 Starting API Integration Test Suite');
    console.log('=' .repeat(60));

    // Phase 1: System Health and Basic Endpoints
    console.log('\n📡 Phase 1: System Health Tests');
    
    const healthTest = await this.testEndpoint('/health');
    if (healthTest.success) {
      this.testResults.systemHealth = healthTest.data;
    }

    await this.testEndpoint('/api/v1/system/health', 'GET', null, [200, 404]);

    // Phase 2: Market Data Integration Tests
    console.log('\n📊 Phase 2: Market Data Integration Tests');
    
    const dataHealthTest = await this.testEndpoint('/api/v1/nautilus-data/health');
    if (dataHealthTest.success && Array.isArray(dataHealthTest.data)) {
      dataHealthTest.data.forEach(source => {
        this.testResults.dataSourceStatus[source.source] = source.status;
      });
    }

    // Test data source endpoints
    await this.testEndpoint('/api/v1/nautilus-data/fred/macro-factors');
    await this.testEndpoint('/api/v1/nautilus-data/alpha-vantage/search?keywords=AAPL');
    await this.testEndpoint('/api/v1/edgar/companies/search?q=Apple');
    
    // Phase 3: Engine-Specific Tests
    console.log('\n🔧 Phase 3: Engine-Specific Tests');
    
    // Risk Engine tests
    await this.testEndpoint('/api/v1/enhanced-risk/health', 'GET', null, [200, 404]);
    await this.testEndpoint('/api/v1/enhanced-risk/system/metrics', 'GET', null, [200, 404]);
    await this.testEndpoint('/api/v1/enhanced-risk/dashboard/views', 'GET', null, [200, 404]);

    // Volatility Engine tests
    await this.testEndpoint('/api/v1/volatility/health');
    await this.testEndpoint('/api/v1/volatility/status');
    await this.testEndpoint('/api/v1/volatility/models');
    await this.testEndpoint('/api/v1/volatility/streaming/status');

    // Phase 4: Portfolio and Trading Tests
    console.log('\n💼 Phase 4: Portfolio and Trading Tests');
    
    await this.testEndpoint('/api/v1/portfolio/positions');
    await this.testEndpoint('/api/v1/portfolio/balance');
    await this.testEndpoint('/api/v1/exchanges/status');

    // Phase 5: M4 Max Hardware Acceleration Tests
    console.log('\n⚡ Phase 5: Hardware Acceleration Tests');
    
    await this.testEndpoint('/api/v1/monitoring/m4max/hardware/metrics');
    await this.testEndpoint('/api/v1/optimization/health');
    await this.testEndpoint('/api/v1/optimization/core-utilization');
    await this.testEndpoint('/api/v1/monitoring/containers/metrics');

    // Phase 6: Performance Tests with Data Operations
    console.log('\n🏃 Phase 6: Data Operation Performance Tests');
    
    // Test POST endpoints with sample data
    await this.testEndpoint('/api/v1/enhanced-risk/backtest/run', 'POST', {
      symbol: 'AAPL',
      start_date: '2023-01-01',
      end_date: '2023-12-31',
      initial_cash: 100000
    }, [200, 422, 404]);

    await this.testEndpoint('/api/v1/volatility/symbols/AAPL/add', 'POST', {
      symbol: 'AAPL'
    }, [200, 422, 404]);

    // Generate comprehensive results
    this.analyzeResults();
    this.printResults();
    
    return this.testResults;
  }

  analyzeResults() {
    const successful = this.testResults.endpointTests.filter(r => r.success).length;
    const total = this.testResults.endpointTests.length;
    const successRate = (successful / total) * 100;

    // Categorize endpoints by response time
    const fastEndpoints = this.testResults.endpointTests.filter(r => r.success && r.responseTime < 100);
    const moderateEndpoints = this.testResults.endpointTests.filter(r => r.success && r.responseTime >= 100 && r.responseTime < 1000);
    const slowEndpoints = this.testResults.endpointTests.filter(r => r.success && r.responseTime >= 1000);

    // Identify data-rich endpoints
    const dataRichEndpoints = this.testResults.endpointTests.filter(r => r.success && r.hasData && r.dataSize > 1000);

    this.testResults.analysis = {
      successRate,
      successful,
      total,
      performanceCategories: {
        fast: fastEndpoints.length,
        moderate: moderateEndpoints.length,
        slow: slowEndpoints.length
      },
      dataRichEndpoints: dataRichEndpoints.length,
      operationalDataSources: Object.values(this.testResults.dataSourceStatus).filter(status => status === 'operational').length,
      totalDataSources: Object.keys(this.testResults.dataSourceStatus).length
    };
  }

  printResults() {
    console.log('\n📊 API Integration Test Results');
    console.log('=' .repeat(60));
    
    const analysis = this.testResults.analysis;
    
    console.log(`🎯 Overall Success Rate: ${analysis.successful}/${analysis.total} (${analysis.successRate.toFixed(1)}%)`);
    console.log(`⚡ Performance Metrics:`);
    console.log(`   Average Response Time: ${this.testResults.performanceMetrics.averageResponseTime.toFixed(1)}ms`);
    console.log(`   Min Response Time: ${this.testResults.performanceMetrics.minResponseTime.toFixed(1)}ms`);
    console.log(`   Max Response Time: ${this.testResults.performanceMetrics.maxResponseTime.toFixed(1)}ms`);
    console.log(`   Timeouts: ${this.testResults.performanceMetrics.timeouts}`);

    console.log(`📊 Data Sources:`);
    console.log(`   Operational: ${analysis.operationalDataSources}/${analysis.totalDataSources}`);
    Object.entries(this.testResults.dataSourceStatus).forEach(([source, status]) => {
      const icon = status === 'operational' ? '✅' : '❌';
      console.log(`   ${icon} ${source}: ${status}`);
    });

    console.log(`🚀 Performance Categories:`);
    console.log(`   Fast (<100ms): ${analysis.performanceCategories.fast} endpoints`);
    console.log(`   Moderate (100-1000ms): ${analysis.performanceCategories.moderate} endpoints`);
    console.log(`   Slow (>1000ms): ${analysis.performanceCategories.slow} endpoints`);
    console.log(`   Data-Rich Endpoints: ${analysis.dataRichEndpoints}`);

    if (this.testResults.systemHealth) {
      console.log(`🏥 System Health: ${this.testResults.systemHealth.status || 'Unknown'}`);
    }

    // Show top performing and problematic endpoints
    const sortedBySpeed = this.testResults.endpointTests
      .filter(r => r.success)
      .sort((a, b) => a.responseTime - b.responseTime);
    
    if (sortedBySpeed.length > 0) {
      console.log(`\n⚡ Top 5 Fastest Endpoints:`);
      sortedBySpeed.slice(0, 5).forEach((result, index) => {
        console.log(`   ${index + 1}. ${result.testName}: ${result.responseTime}ms`);
      });
    }

    const failures = this.testResults.endpointTests.filter(r => !r.success);
    if (failures.length > 0) {
      console.log(`\n❌ Failed Endpoints (${failures.length}):`);
      failures.forEach((result, index) => {
        console.log(`   ${index + 1}. ${result.testName}: ${result.error}`);
      });
    }

    // Integration Assessment
    console.log('\n🎯 Frontend-Backend Integration Assessment:');
    if (analysis.successRate >= 85) {
      console.log('   ✅ EXCELLENT: Frontend-backend integration is robust');
    } else if (analysis.successRate >= 70) {
      console.log('   ✅ GOOD: Most API endpoints are functioning well');
    } else if (analysis.successRate >= 50) {
      console.log('   ⚠️ PARTIAL: Some integration issues need attention');
    } else {
      console.log('   ❌ POOR: Significant integration problems detected');
    }

    if (this.testResults.performanceMetrics.averageResponseTime < 200) {
      console.log('   ⚡ PERFORMANCE: Excellent response times');
    } else if (this.testResults.performanceMetrics.averageResponseTime < 500) {
      console.log('   ⚡ PERFORMANCE: Good response times');
    } else {
      console.log('   ⚠️ PERFORMANCE: Response times could be improved');
    }
  }
}

// Run the test suite
async function main() {
  const tester = new APIIntegrationTester();
  
  try {
    await tester.runFullTestSuite();
    
    const analysis = tester.testResults.analysis;
    
    if (analysis.successRate >= 70) {
      console.log('\n🎉 API integration tests completed successfully!');
      process.exit(0);
    } else {
      console.log('\n⚠️ API integration tests completed with significant issues.');
      process.exit(1);
    }
    
  } catch (error) {
    console.error('\n💥 Test suite failed:', error.message);
    process.exit(1);
  }
}

// Add node-fetch dependency check
async function checkDependencies() {
  try {
    await import('node-fetch');
    return true;
  } catch {
    console.error('❌ Missing dependency: node-fetch');
    console.log('📦 Please install: npm install node-fetch');
    return false;
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  checkDependencies().then(hasDepends => {
    if (hasDepends) {
      main();
    } else {
      process.exit(1);
    }
  });
}

export default APIIntegrationTester;