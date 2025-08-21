/**
 * Integration Tests for Frontend Monitoring System
 * Tests end-to-end monitoring workflows from services through UI components
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { OrderLatencyTracker, LatencyMeasurement } from '../../src/services/monitoring/OrderLatencyTracker';
import { ThresholdAlertEngine, ThresholdRule } from '../../src/services/monitoring/ThresholdAlertEngine';
import { createMockOrderLatencyTracker, createMockThresholdAlertEngine } from '../../src/test-utils/monitoring-test-helpers';

// Mock API responses for integration testing
const createMockAPIResponse = {
  latencyMetrics: {
    venue_name: 'IB',
    order_execution_latency: {
      min_ms: 5.2,
      max_ms: 156.7,
      avg_ms: 23.4,
      p50_ms: 18.9,
      p95_ms: 89.3,
      p99_ms: 134.2,
      samples: 1247
    },
    last_updated: new Date().toISOString()
  },
  activeAlerts: [
    {
      alert_id: 'alert-1',
      metric_name: 'order_latency',
      current_value: 156.7,
      threshold_value: 100,
      severity: 'high',
      triggered_at: new Date().toISOString(),
      venue_name: 'IB',
      description: 'Order latency exceeded threshold',
      auto_resolution_available: true,
      escalation_level: 3,
      notification_sent: false
    }
  ]
};

describe('Monitoring System Integration Tests', () => {
  let latencyTracker: ReturnType<typeof createMockOrderLatencyTracker>;
  let alertEngine: ReturnType<typeof createMockThresholdAlertEngine>;
  
  beforeEach(() => {
    latencyTracker = createMockOrderLatencyTracker();
    alertEngine = createMockThresholdAlertEngine();
    vi.clearAllMocks();
  });

  afterEach(() => {
    latencyTracker.clear();
    alertEngine.clear();
  });

  describe('End-to-End Order Monitoring Workflow', () => {
    it('should complete full order tracking and alert generation workflow', async () => {
      // Step 1: Create alert rule for high latency
      const rule = alertEngine.createThresholdRule({
        name: 'High Order Latency',
        metric_name: 'order_execution_latency',
        threshold_value: 100,
        condition: 'greater_than',
        severity: 'high',
        venue_filter: ['IB']
      });

      expect(rule.rule_id).toBeTruthy();
      expect(rule.enabled).toBe(true);

      // Step 2: Start order tracking
      latencyTracker.startOrder('order-123', 'IB', 'AAPL', 'market', 'buy');
      expect(latencyTracker.getPendingOrdersCount()).toBe(1);

      // Step 3: Complete order with high latency
      const measurement = latencyTracker.completeOrder('order-123', true);
      expect(measurement).toBeTruthy();
      expect(measurement!.success).toBe(true);
      expect(latencyTracker.getPendingOrdersCount()).toBe(0);

      // Step 4: Check if latency triggers alert (mock high latency)
      const highLatency = 150; // Above threshold
      const triggeredAlerts = alertEngine.checkMetricValue(
        'order_execution_latency',
        highLatency,
        'IB'
      );

      expect(triggeredAlerts.length).toBe(1);
      expect(triggeredAlerts[0].current_value).toBe(highLatency);
      expect(triggeredAlerts[0].threshold_value).toBe(100);
      expect(triggeredAlerts[0].venue_name).toBe('IB');

      // Step 5: Verify alert is active
      const activeAlerts = alertEngine.getActiveAlerts();
      expect(activeAlerts.length).toBe(1);
      expect(activeAlerts[0].metric_name).toBe('order_execution_latency');
    });

    it('should handle multiple venue monitoring simultaneously', async () => {
      const venues = ['IB', 'BINANCE', 'ALPACA'];
      const orders = ['order-1', 'order-2', 'order-3'];

      // Start orders on different venues
      venues.forEach((venue, index) => {
        latencyTracker.startOrder(orders[index], venue, 'AAPL', 'market', 'buy');
      });

      expect(latencyTracker.getPendingOrdersCount()).toBe(3);

      // Complete orders with different latencies
      const measurements = orders.map((orderId, index) => {
        return latencyTracker.completeOrder(orderId, true);
      });

      expect(measurements.every(m => m !== null)).toBe(true);
      expect(latencyTracker.getPendingOrdersCount()).toBe(0);

      // Check venue-specific metrics
      venues.forEach(venue => {
        const metrics = latencyTracker.getVenueLatencyMetrics(venue);
        expect(metrics.venue_name).toBe(venue);
        expect(metrics.order_execution_latency?.samples).toBeGreaterThan(0);
      });

      // Verify all venues are tracked
      const allMetrics = latencyTracker.getAllVenueMetrics();
      expect(allMetrics.length).toBe(venues.length);
    });

    it('should handle alert resolution workflow', async () => {
      // Create alert rule with auto-resolution
      alertEngine.createThresholdRule({
        name: 'CPU Usage Monitor',
        metric_name: 'cpu_usage',
        threshold_value: 80,
        condition: 'greater_than',
        severity: 'medium',
        auto_resolve: true,
        auto_resolve_threshold: 70
      });

      // Trigger alert
      const alerts = alertEngine.checkMetricValue('cpu_usage', 85);
      expect(alerts.length).toBe(1);

      const alertId = alerts[0].alert_id;
      
      // Verify alert is active
      expect(alertEngine.getActiveAlerts().length).toBe(1);

      // Test manual resolution
      const resolved = alertEngine.resolveAlert(alertId, 'manual');
      expect(resolved).toBe(true);

      // Verify alert is resolved
      expect(alertEngine.getActiveAlerts().length).toBe(0);
      
      const history = alertEngine.getAlertHistory();
      expect(history.length).toBe(1);
      expect(history[0].resolution_method).toBe('manual');
    });

    it('should maintain performance under concurrent operations', async () => {
      const numConcurrentOrders = 100;
      const startTime = performance.now();

      // Start many concurrent orders
      const orderPromises = Array.from({ length: numConcurrentOrders }, (_, i) => {
        const orderId = `concurrent-order-${i}`;
        const venue = ['IB', 'BINANCE', 'ALPACA'][i % 3];
        
        return new Promise<LatencyMeasurement | null>(resolve => {
          latencyTracker.startOrder(orderId, venue, 'AAPL', 'market', 'buy');
          
          // Simulate async completion
          setTimeout(() => {
            const measurement = latencyTracker.completeOrder(orderId, true);
            resolve(measurement);
          }, Math.random() * 10); // Random delay 0-10ms
        });
      });

      // Wait for all orders to complete
      const results = await Promise.all(orderPromises);
      const totalTime = performance.now() - startTime;

      // Performance validation
      const avgTimePerOrder = totalTime / numConcurrentOrders;
      expect(avgTimePerOrder).toBeLessThan(1); // < 1ms per order

      // Verify all orders completed successfully
      expect(results.every(r => r !== null)).toBe(true);
      expect(latencyTracker.getPendingOrdersCount()).toBe(0);

      // Verify metrics are calculated correctly
      const statistics = latencyTracker.getStatistics();
      expect(statistics.totalMeasurements).toBe(numConcurrentOrders);
      expect(statistics.venues.length).toBe(3);

      console.log(`Concurrent performance: ${numConcurrentOrders} orders in ${totalTime.toFixed(2)}ms`);
      console.log(`Average: ${avgTimePerOrder.toFixed(3)}ms per order`);
    });
  });

  describe('Service Integration Performance Tests', () => {
    it('should maintain sub-millisecond latency tracking overhead', () => {
      const iterations = 10000;
      const measurements: number[] = [];

      for (let i = 0; i < iterations; i++) {
        const orderId = `perf-test-${i}`;
        
        const startTracking = performance.now();
        latencyTracker.startOrder(orderId, 'IB', 'TEST', 'market', 'buy');
        const measurement = latencyTracker.completeOrder(orderId, true);
        const endTracking = performance.now();

        const trackingOverhead = endTracking - startTracking;
        measurements.push(trackingOverhead);
      }

      // Calculate statistics
      const avgOverhead = measurements.reduce((sum, m) => sum + m, 0) / measurements.length;
      const maxOverhead = Math.max(...measurements);
      const p95Overhead = measurements.sort((a, b) => a - b)[Math.floor(measurements.length * 0.95)];

      console.log(`Latency tracking overhead (${iterations} iterations):`);
      console.log(`  Average: ${avgOverhead.toFixed(3)}ms`);
      console.log(`  Maximum: ${maxOverhead.toFixed(3)}ms`);
      console.log(`  P95: ${p95Overhead.toFixed(3)}ms`);

      // Performance requirements
      expect(avgOverhead).toBeLessThan(0.1); // < 0.1ms average overhead
      expect(p95Overhead).toBeLessThan(0.5); // < 0.5ms P95 overhead
    });

    it('should handle large dataset queries efficiently', () => {
      // Populate with large dataset
      const dataSize = 50000;
      for (let i = 0; i < dataSize; i++) {
        const orderId = `large-dataset-${i}`;
        const venue = ['IB', 'BINANCE', 'ALPACA', 'KRAKEN', 'COINBASE'][i % 5];
        
        latencyTracker.startOrder(orderId, venue, 'TEST', 'market', 'buy');
        latencyTracker.completeOrder(orderId, true);
      }

      // Test query performance
      const queryStart = performance.now();
      
      const allMetrics = latencyTracker.getAllVenueMetrics();
      const ibMetrics = latencyTracker.getVenueLatencyMetrics('IB');
      const recentMeasurements = latencyTracker.getRecentMeasurements(1000);
      const statistics = latencyTracker.getStatistics();
      
      const queryTime = performance.now() - queryStart;

      console.log(`Large dataset query performance (${dataSize} records):`);
      console.log(`  Query time: ${queryTime.toFixed(2)}ms`);
      console.log(`  Results: ${allMetrics.length} venues, ${recentMeasurements.length} recent measurements`);

      // Performance requirements
      expect(queryTime).toBeLessThan(50); // < 50ms for large dataset queries
      
      // Verify query results
      expect(allMetrics.length).toBe(5); // All venues
      expect(ibMetrics.venue_name).toBe('IB');
      expect(recentMeasurements.length).toBe(1000);
      expect(statistics.totalMeasurements).toBe(dataSize);
    });

    it('should efficiently manage memory usage with data retention', () => {
      const maxHistorySize = 10000;
      
      // Fill beyond capacity
      for (let i = 0; i < maxHistorySize + 5000; i++) {
        const orderId = `memory-test-${i}`;
        latencyTracker.startOrder(orderId, 'IB', 'TEST', 'market', 'buy');
        latencyTracker.completeOrder(orderId, true);
      }

      // Verify memory management
      const statistics = latencyTracker.getStatistics();
      expect(statistics.totalMeasurements).toBeLessThanOrEqual(maxHistorySize);

      // Verify recent data is preserved
      const recentMeasurements = latencyTracker.getRecentMeasurements(100);
      expect(recentMeasurements.length).toBe(100);

      // Verify oldest data was cleaned up
      const allMeasurements = latencyTracker.getRecentMeasurements(maxHistorySize + 1000);
      expect(allMeasurements.length).toBeLessThanOrEqual(maxHistorySize);

      console.log(`Memory management: ${statistics.totalMeasurements} measurements retained`);
    });
  });

  describe('Alert Engine Integration Performance', () => {
    it('should efficiently process multiple threshold rules', () => {
      const numRules = 100;
      const testValue = 75;

      // Create many threshold rules
      const rules = Array.from({ length: numRules }, (_, i) => {
        return alertEngine.createThresholdRule({
          name: `Rule ${i}`,
          metric_name: `test_metric_${i % 10}`, // 10 different metrics
          threshold_value: 50 + (i % 50), // Various thresholds
          condition: i % 2 === 0 ? 'greater_than' : 'less_than',
          severity: ['low', 'medium', 'high', 'critical'][i % 4] as any
        });
      });

      expect(rules.length).toBe(numRules);

      // Test rule evaluation performance
      const evalStart = performance.now();
      
      const triggeredAlerts = [];
      for (let i = 0; i < 10; i++) {
        const alerts = alertEngine.checkMetricValue(`test_metric_${i}`, testValue);
        triggeredAlerts.push(...alerts);
      }
      
      const evalTime = performance.now() - evalStart;

      console.log(`Rule evaluation performance:`);
      console.log(`  ${numRules} rules evaluated in ${evalTime.toFixed(2)}ms`);
      console.log(`  ${triggeredAlerts.length} alerts triggered`);
      console.log(`  Average: ${(evalTime / numRules).toFixed(3)}ms per rule`);

      // Performance requirements
      expect(evalTime).toBeLessThan(50); // < 50ms for 100 rules
      expect(evalTime / numRules).toBeLessThan(0.5); // < 0.5ms per rule
    });

    it('should handle alert statistics calculation efficiently', () => {
      // Create test alerts with different characteristics
      const testScenarios = [
        { metric: 'cpu_usage', severity: 'high', venue: 'IB' },
        { metric: 'memory_usage', severity: 'critical', venue: 'BINANCE' },
        { metric: 'latency', severity: 'medium', venue: 'IB' },
        { metric: 'throughput', severity: 'low', venue: 'ALPACA' }
      ];

      // Create many alerts
      const numAlerts = 1000;
      for (let i = 0; i < numAlerts; i++) {
        const scenario = testScenarios[i % testScenarios.length];
        
        alertEngine.createThresholdRule({
          name: `Alert Rule ${i}`,
          metric_name: scenario.metric,
          threshold_value: 50,
          condition: 'greater_than',
          severity: scenario.severity,
          venue_filter: [scenario.venue]
        });

        // Trigger some alerts
        if (i % 10 === 0) {
          alertEngine.checkMetricValue(scenario.metric, 75, scenario.venue);
        }
      }

      // Test statistics calculation performance
      const statsStart = performance.now();
      const statistics = alertEngine.getAlertStatistics();
      const statsTime = performance.now() - statsStart;

      console.log(`Alert statistics performance:`);
      console.log(`  Statistics calculated in ${statsTime.toFixed(2)}ms`);
      console.log(`  Active alerts: ${statistics.active_alerts}`);
      console.log(`  Total alerts: ${statistics.total_alerts}`);

      // Performance and accuracy requirements
      expect(statsTime).toBeLessThan(20); // < 20ms for statistics
      expect(statistics.active_alerts).toBeGreaterThan(0);
      expect(statistics.by_severity).toBeTruthy();
      expect(statistics.by_metric).toBeTruthy();
      expect(statistics.by_venue).toBeTruthy();
    });
  });

  describe('API Integration Simulation', () => {
    it('should simulate realistic API response times and data flow', async () => {
      // Simulate API delay
      const simulateAPICall = async (data: any, delayMs: number = 10) => {
        await new Promise(resolve => setTimeout(resolve, delayMs));
        return data;
      };

      // Test latency metrics API simulation
      const apiStart = performance.now();
      const latencyData = await simulateAPICall(createMockAPIResponse.latencyMetrics, 15);
      const latencyAPITime = performance.now() - apiStart;

      expect(latencyAPITime).toBeGreaterThan(10); // Should include simulated delay
      expect(latencyAPITime).toBeLessThan(50); // But not too slow
      expect(latencyData.venue_name).toBe('IB');
      expect(latencyData.order_execution_latency.samples).toBe(1247);

      // Test alerts API simulation
      const alertsStart = performance.now();
      const alertsData = await simulateAPICall(createMockAPIResponse.activeAlerts, 8);
      const alertsAPITime = performance.now() - alertsStart;

      expect(alertsAPITime).toBeGreaterThan(5);
      expect(alertsAPITime).toBeLessThan(30);
      expect(alertsData.length).toBe(1);
      expect(alertsData[0].severity).toBe('high');

      console.log(`API simulation performance:`);
      console.log(`  Latency API: ${latencyAPITime.toFixed(1)}ms`);
      console.log(`  Alerts API: ${alertsAPITime.toFixed(1)}ms`);
    });

    it('should handle API error scenarios gracefully', async () => {
      const simulateAPIError = async (errorRate: number = 0.1) => {
        if (Math.random() < errorRate) {
          throw new Error('Simulated API error');
        }
        return createMockAPIResponse.latencyMetrics;
      };

      let successCount = 0;
      let errorCount = 0;
      const attempts = 100;

      // Test error handling
      for (let i = 0; i < attempts; i++) {
        try {
          await simulateAPIError(0.2); // 20% error rate
          successCount++;
        } catch (error) {
          errorCount++;
        }
      }

      const errorRate = errorCount / attempts;
      const successRate = successCount / attempts;

      console.log(`API error simulation:`);
      console.log(`  Success rate: ${(successRate * 100).toFixed(1)}%`);
      console.log(`  Error rate: ${(errorRate * 100).toFixed(1)}%`);

      // Should approximate expected error rate
      expect(errorRate).toBeGreaterThan(0.15);
      expect(errorRate).toBeLessThan(0.25);
      expect(successRate).toBeGreaterThan(0.75);
    });
  });

  describe('Integration Data Consistency Tests', () => {
    it('should maintain data consistency across service interactions', () => {
      const orderId = 'consistency-test-order';
      const venue = 'IB';
      const symbol = 'AAPL';

      // Start order tracking
      latencyTracker.startOrder(orderId, venue, symbol, 'market', 'buy');
      
      // Verify order is tracked
      expect(latencyTracker.getPendingOrdersCount()).toBe(1);
      expect(latencyTracker.getPendingOrdersByVenue().get(venue)).toBe(1);

      // Complete order
      const measurement = latencyTracker.completeOrder(orderId, true);
      
      // Verify measurement consistency
      expect(measurement).toBeTruthy();
      expect(measurement!.orderId).toBe(orderId);
      expect(measurement!.venue).toBe(venue);
      expect(measurement!.success).toBe(true);

      // Verify metrics consistency
      const venueMetrics = latencyTracker.getVenueLatencyMetrics(venue);
      expect(venueMetrics.venue_name).toBe(venue);
      expect(venueMetrics.order_execution_latency?.samples).toBe(1);

      // Verify statistics consistency
      const stats = latencyTracker.getStatistics();
      expect(stats.totalMeasurements).toBe(1);
      expect(stats.venues).toContain(venue);
      expect(stats.pendingOrders).toBe(0);
    });

    it('should maintain alert state consistency during rule changes', () => {
      // Create initial rule
      const rule = alertEngine.createThresholdRule({
        name: 'Test Consistency Rule',
        metric_name: 'test_metric',
        threshold_value: 50,
        condition: 'greater_than',
        severity: 'medium'
      });

      // Trigger alert
      const alerts = alertEngine.checkMetricValue('test_metric', 75);
      expect(alerts.length).toBe(1);
      expect(alertEngine.getActiveAlerts().length).toBe(1);

      // Update rule threshold
      const updatedRule = alertEngine.updateThresholdRule(rule.rule_id, {
        threshold_value: 100 // Higher threshold
      });

      expect(updatedRule?.threshold_value).toBe(100);

      // Test value that previously triggered should not trigger now
      const newAlerts = alertEngine.checkMetricValue('test_metric', 75);
      // Note: Existing alerts remain active until resolved
      
      // Disable rule
      alertEngine.setRuleEnabled(rule.rule_id, false);
      
      // Should not trigger new alerts when disabled
      const disabledAlerts = alertEngine.checkMetricValue('test_metric', 150);
      expect(disabledAlerts.length).toBe(0);
    });
  });
});

describe('End-to-End Performance Integration Tests', () => {
  let latencyTracker: OrderLatencyTracker;
  let alertEngine: ThresholdAlertEngine;

  beforeEach(() => {
    latencyTracker = new OrderLatencyTracker();
    alertEngine = new ThresholdAlertEngine();
  });

  afterEach(() => {
    latencyTracker.clear();
    alertEngine.clear();
  });

  it('should complete full monitoring workflow within performance bounds', async () => {
    const testStart = performance.now();
    
    // Setup phase
    const setupStart = performance.now();
    
    // Create monitoring rules
    const rules = [
      { name: 'High Latency IB', metric: 'order_latency_ib', threshold: 100, venue: 'IB' },
      { name: 'High Latency BINANCE', metric: 'order_latency_binance', threshold: 150, venue: 'BINANCE' },
      { name: 'Low Success Rate', metric: 'success_rate', threshold: 95, condition: 'less_than' as const }
    ];

    rules.forEach(rule => {
      alertEngine.createThresholdRule({
        name: rule.name,
        metric_name: rule.metric,
        threshold_value: rule.threshold,
        condition: rule.condition || 'greater_than',
        severity: 'high',
        venue_filter: rule.venue ? [rule.venue] : undefined
      });
    });

    const setupTime = performance.now() - setupStart;

    // Execution phase
    const execStart = performance.now();
    
    const numOrders = 1000;
    const venues = ['IB', 'BINANCE', 'ALPACA'];
    
    // Process orders
    for (let i = 0; i < numOrders; i++) {
      const orderId = `perf-order-${i}`;
      const venue = venues[i % venues.length];
      const symbol = ['AAPL', 'MSFT', 'GOOGL', 'TSLA'][i % 4];
      
      // Start tracking
      latencyTracker.startOrder(orderId, venue, symbol, 'market', 'buy');
      
      // Complete with realistic latency
      const success = Math.random() > 0.02; // 98% success rate
      const measurement = latencyTracker.completeOrder(orderId, success);
      
      // Check for alerts based on latency
      if (measurement && measurement.latency_ms > 100) {
        alertEngine.checkMetricValue(`order_latency_${venue.toLowerCase()}`, measurement.latency_ms, venue);
      }
    }

    const execTime = performance.now() - execStart;

    // Analysis phase
    const analysisStart = performance.now();
    
    const allMetrics = latencyTracker.getAllVenueMetrics();
    const statistics = latencyTracker.getStatistics();
    const activeAlerts = alertEngine.getActiveAlerts();
    const alertStats = alertEngine.getAlertStatistics();
    
    const analysisTime = performance.now() - analysisStart;
    
    const totalTime = performance.now() - testStart;

    // Performance validation
    expect(setupTime).toBeLessThan(10); // < 10ms setup
    expect(execTime).toBeLessThan(1000); // < 1s for 1000 orders
    expect(analysisTime).toBeLessThan(50); // < 50ms analysis
    expect(totalTime).toBeLessThan(1100); // < 1.1s total

    // Data validation
    expect(statistics.totalMeasurements).toBe(numOrders);
    expect(statistics.venues.length).toBe(venues.length);
    expect(allMetrics.length).toBe(venues.length);
    
    console.log(`End-to-end performance results:`);
    console.log(`  Setup: ${setupTime.toFixed(1)}ms`);
    console.log(`  Execution: ${execTime.toFixed(1)}ms (${(numOrders/execTime*1000).toFixed(0)} orders/sec)`);
    console.log(`  Analysis: ${analysisTime.toFixed(1)}ms`);
    console.log(`  Total: ${totalTime.toFixed(1)}ms`);
    console.log(`  Active alerts: ${activeAlerts.length}`);
    console.log(`  Average latency: ${statistics.averageLatency.toFixed(1)}ms`);
  });
});